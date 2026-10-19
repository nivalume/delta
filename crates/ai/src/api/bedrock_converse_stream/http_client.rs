//! HTTP clients handed to the SDK, the counterparts of upstream's request handlers:
//!
//! - [`direct_https_client`]: the SDK's default HTTPS client (hyper, rustls with aws-lc, HTTP/2
//!   negotiated through ALPN) without its process-environment proxy lookup. Upstream's default
//!   `NodeHttp2Handler` connects directly; proxies are decided by [`super::proxy`] alone.
//! - [`Http1Client`]: HTTP/1.1 only, optionally through one HTTP(S) proxy for every request. It
//!   stands in for upstream's `NodeHttpHandler` (with `HttpProxyAgent`/`HttpsProxyAgent` when a
//!   proxy applies), and runs on `reqwest` because the SDK's own client cannot restrict ALPN to
//!   `http/1.1`.

use std::fmt;

use aws_sdk_bedrockruntime::config::http::{HttpRequest, HttpResponse};
use aws_sdk_bedrockruntime::config::{HttpClient, RuntimeComponents, SharedHttpClient};
use aws_smithy_http_client::tls::{self, rustls_provider::CryptoMode};
use aws_smithy_runtime_api::client::http::{
    HttpConnector, HttpConnectorFuture, HttpConnectorSettings, SharedHttpConnector,
};
use aws_smithy_runtime_api::client::result::ConnectorError;
use aws_smithy_types::body::SdkBody;
use url::Url;

/// The SDK's default HTTPS client with proxies disabled.
pub(super) fn direct_https_client() -> SharedHttpClient {
    // Without a proxy config the connector builder uses `ProxyConfig::disabled()`; the SDK's
    // default client sets `ProxyConfig::from_env()` instead.
    aws_smithy_http_client::Builder::new()
        .tls_provider(tls::Provider::Rustls(CryptoMode::AwsLc))
        .build_https()
}

/// HTTP/1.1-only client; `proxy` carries every request when set.
#[derive(Clone, PartialEq, Eq)]
pub(super) struct Http1Client {
    proxy: Option<Url>,
}

impl Http1Client {
    pub(super) fn new(proxy: Option<Url>) -> Self {
        Self { proxy }
    }

    fn reqwest_client(&self, settings: &HttpConnectorSettings) -> reqwest::Result<reqwest::Client> {
        let mut builder = reqwest::Client::builder()
            .http1_only()
            // Node's HTTP handler does not follow redirects; the SDK sees the 3xx response.
            .redirect(reqwest::redirect::Policy::none());
        builder = match &self.proxy {
            // Like an agent, the proxy carries every request; `NO_PROXY` was applied when the
            // proxy was resolved.
            Some(proxy) => builder.proxy(reqwest::Proxy::all(proxy.clone())?),
            None => builder.no_proxy(),
        };
        if let Some(timeout) = settings.connect_timeout() {
            builder = builder.connect_timeout(timeout);
        }
        builder.build()
    }
}

impl fmt::Debug for Http1Client {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Http1Client")
            .field("proxy", &self.proxy.as_ref().map(redacted))
            .finish()
    }
}

/// `scheme://host:port` of a proxy URL, without credentials or path.
pub(super) fn redacted(url: &Url) -> String {
    format!(
        "{}://{}:{}",
        url.scheme(),
        url.host_str().unwrap_or_default(),
        url.port_or_known_default().unwrap_or_default()
    )
}

impl HttpClient for Http1Client {
    fn http_connector(&self, settings: &HttpConnectorSettings, _: &RuntimeComponents) -> SharedHttpConnector {
        SharedHttpConnector::new(Http1Connector {
            client: self.reqwest_client(settings).map_err(|error| error.to_string()),
            read_timeout: settings.read_timeout(),
        })
    }
}

#[derive(Debug)]
struct Http1Connector {
    client: std::result::Result<reqwest::Client, String>,
    /// Time allowed until the response head arrives, matching the SDK's hyper connector (the
    /// body is not covered).
    read_timeout: Option<std::time::Duration>,
}

fn into_reqwest_body(body: SdkBody) -> reqwest::Body {
    match body.bytes() {
        Some(bytes) => reqwest::Body::from(bytes::Bytes::copy_from_slice(bytes)),
        None => reqwest::Body::wrap(body),
    }
}

fn connector_error(error: reqwest::Error) -> ConnectorError {
    if error.is_timeout() {
        ConnectorError::timeout(error.into())
    } else if error.is_builder() {
        ConnectorError::user(error.into())
    } else {
        // Connection failures, resets and proxy refusals are transient, as for the SDK's own
        // client.
        ConnectorError::io(error.into())
    }
}

impl HttpConnector for Http1Connector {
    fn call(&self, request: HttpRequest) -> HttpConnectorFuture {
        let client = match &self.client {
            Ok(client) => client.clone(),
            Err(message) => return HttpConnectorFuture::ready(Err(ConnectorError::user(message.clone().into()))),
        };
        let request = match request.try_into_http1x() {
            Ok(request) => request.map(into_reqwest_body),
            Err(error) => return HttpConnectorFuture::ready(Err(ConnectorError::user(error.into()))),
        };
        let request = match reqwest::Request::try_from(request) {
            Ok(request) => request,
            Err(error) => return HttpConnectorFuture::ready(Err(ConnectorError::user(error.into()))),
        };
        let read_timeout = self.read_timeout;
        HttpConnectorFuture::new(async move {
            let send = client.execute(request);
            let response = match read_timeout {
                Some(timeout) => tokio::time::timeout(timeout, send)
                    .await
                    .map_err(|elapsed| ConnectorError::timeout(elapsed.into()))?,
                None => send.await,
            }
            .map_err(connector_error)?;
            let response = http::Response::from(response).map(SdkBody::from_body_1_x);
            HttpResponse::try_from(response).map_err(|error| ConnectorError::other(error.into(), None))
        })
    }
}
