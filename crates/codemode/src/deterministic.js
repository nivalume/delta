// Deterministic runs (`Sandbox::start`): evaluated as `codemode-deterministic.js` before the
// prelude, which then freezes what this installs. Upstream fixes time, randomness and the
// timezone at the WASI level; a native QuickJS reads the host's clock and timezone, so this
// replaces what reads them:
// - `Date.now()`, `new Date()`, `Date()` and `performance.now()` read a virtual clock that the
//   host sets before each result (`setNow`), never going backwards;
// - `Math.random()` is mulberry32 over the seed;
// - the local-time `Date` methods, `getTimezoneOffset`, the string forms, `Date` built from
//   components and date-time strings without a zone behave as UTC;
// - `WeakRef` and `FinalizationRegistry` are removed, so collection timing is not observable.
// Evaluates to `(startedAt, seed) => { setNow }`.
(function (startedAt, seed) {
	"use strict";

	let now = startedAt;
	const RealDate = Date;
	const proto = RealDate.prototype;
	const realParse = RealDate.parse;
	const realUTC = RealDate.UTC;
	const realOffset = proto.getTimezoneOffset;
	const getTime = proto.getTime;

	// A date-time string without a zone is local time to the engine: read it as UTC instead.
	const ZONE = /(?:Z|[+-]\d\d(?::?\d\d)?|\b(?:GMT|UTC|UT)(?:[+-]\d{1,4})?)\s*(?:\([^)]*\))?\s*$/i;
	const DATE_ONLY = /^\s*[+-]?\d{4,6}(?:-\d\d(?:-\d\d)?)?\s*$/;
	function parse(text) {
		const string = String(text);
		const value = realParse(string);
		if (value !== value || ZONE.test(string) || DATE_ONLY.test(string)) return value;
		return value - realOffset.call(new RealDate(value)) * 60000;
	}

	function VirtualDate(...args) {
		if (new.target === undefined) return new RealDate(now).toString();
		if (args.length === 0) return Reflect.construct(RealDate, [now], new.target);
		if (args.length === 1) {
			const [value] = args;
			if (typeof value === "string") return Reflect.construct(RealDate, [parse(value)], new.target);
			return Reflect.construct(RealDate, args, new.target);
		}
		return Reflect.construct(RealDate, [realUTC(...args)], new.target);
	}
	Object.defineProperty(VirtualDate, "prototype", { value: proto, writable: false });
	Object.defineProperty(VirtualDate, "name", { value: "Date" });
	Object.defineProperty(VirtualDate, "length", { value: 7 });
	Object.defineProperty(proto, "constructor", { value: VirtualDate, writable: true, configurable: true });
	VirtualDate.now = function now_() {
		return now;
	};
	VirtualDate.parse = parse;
	VirtualDate.UTC = realUTC;

	const define = (name, value) =>
		Object.defineProperty(proto, name, { value, writable: true, enumerable: false, configurable: true });
	for (const field of ["FullYear", "Month", "Date", "Day", "Hours", "Minutes", "Seconds", "Milliseconds"]) {
		define("get" + field, proto["getUTC" + field]);
		if (field !== "Day") define("set" + field, proto["setUTC" + field]);
	}
	define("getTimezoneOffset", function getTimezoneOffset() {
		const value = getTime.call(this);
		return value !== value ? NaN : 0;
	});
	if (typeof proto.getYear === "function") {
		define("getYear", function getYear() {
			return this.getUTCFullYear() - 1900;
		});
	}

	const DAYS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"];
	const MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"];
	const pad = (value, width) => {
		const sign = value < 0 ? "-" : "";
		return sign + String(Math.abs(value)).padStart(width, "0");
	};
	// The engine's own formats with a zone offset of 0.
	function datePart(date) {
		return (
			DAYS[date.getUTCDay()] +
			" " +
			MONTHS[date.getUTCMonth()] +
			" " +
			pad(date.getUTCDate(), 2) +
			" " +
			pad(date.getUTCFullYear(), 4)
		);
	}
	function timePart(date) {
		return (
			pad(date.getUTCHours(), 2) +
			":" +
			pad(date.getUTCMinutes(), 2) +
			":" +
			pad(date.getUTCSeconds(), 2) +
			" GMT+0000"
		);
	}
	function localeDate(date) {
		return pad(date.getUTCMonth() + 1, 2) + "/" + pad(date.getUTCDate(), 2) + "/" + pad(date.getUTCFullYear(), 4);
	}
	function localeTime(date) {
		const hours = date.getUTCHours();
		return (
			pad(((hours + 11) % 12) + 1, 2) +
			":" +
			pad(date.getUTCMinutes(), 2) +
			":" +
			pad(date.getUTCSeconds(), 2) +
			(hours < 12 ? " AM" : " PM")
		);
	}
	const formatted = (format) =>
		function () {
			const value = getTime.call(this);
			return value !== value ? "Invalid Date" : format(this);
		};
	define(
		"toString",
		formatted((date) => datePart(date) + " " + timePart(date)),
	);
	define("toDateString", formatted(datePart));
	define("toTimeString", formatted(timePart));
	define(
		"toLocaleString",
		formatted((date) => localeDate(date) + ", " + localeTime(date)),
	);
	define("toLocaleDateString", formatted(localeDate));
	define("toLocaleTimeString", formatted(localeTime));

	Object.defineProperty(globalThis, "Date", { value: VirtualDate, writable: true, configurable: true });

	// mulberry32: one 32-bit draw per call; 53 bits per Math.random() from two draws.
	let state = seed >>> 0;
	function draw() {
		state = (state + 0x6d2b79f5) >>> 0;
		let value = Math.imul(state ^ (state >>> 15), 1 | state);
		value = (value + Math.imul(value ^ (value >>> 7), 61 | value)) ^ value;
		return (value ^ (value >>> 14)) >>> 0;
	}
	Object.defineProperty(Math, "random", {
		value: function random() {
			return ((draw() >>> 5) * 67108864 + (draw() >>> 6)) / 9007199254740992;
		},
		writable: true,
		configurable: true,
	});

	const performance = Object.create(null);
	Object.defineProperty(performance, "timeOrigin", { value: startedAt, enumerable: true });
	Object.defineProperty(performance, "now", {
		value: function now_() {
			return now - startedAt;
		},
		enumerable: true,
	});
	Object.defineProperty(globalThis, "performance", { value: performance, writable: true, configurable: true });

	delete globalThis.WeakRef;
	delete globalThis.FinalizationRegistry;

	return {
		setNow(value) {
			if (value === value && value > now) now = value;
		},
	};
})
