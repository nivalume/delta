//! Python's `random` module, as much as `random.seed(n)` and `random.shuffle` need: the
//! Mersenne Twister seeded by `init_by_array`, `getrandbits` and `_randbelow`.

/// MT19937 seeded and drawn as CPython's `random.Random`.
#[derive(Clone, Debug)]
pub struct PyRandom {
    state: [u32; 624],
    index: usize,
}

impl PyRandom {
    /// `random.seed(seed)` for a non-negative integer seed below 2^64.
    pub fn new(seed: u64) -> Self {
        let mut key: Vec<u32> = Vec::new();
        let mut rest = seed;
        loop {
            key.push((rest & 0xffff_ffff) as u32);
            rest >>= 32;
            if rest == 0 {
                break;
            }
        }
        let mut random = Self {
            state: [0; 624],
            index: 624,
        };
        random.init_by_array(&key);
        random
    }

    fn init_genrand(&mut self, seed: u32) {
        self.state[0] = seed;
        for i in 1..624 {
            let previous = self.state[i - 1];
            self.state[i] = 1_812_433_253_u32
                .wrapping_mul(previous ^ (previous >> 30))
                .wrapping_add(i as u32);
        }
        self.index = 624;
    }

    fn init_by_array(&mut self, key: &[u32]) {
        self.init_genrand(19_650_218);
        let (mut i, mut j) = (1_usize, 0_usize);
        for _ in 0..624.max(key.len()) {
            let previous = self.state[i - 1];
            self.state[i] = (self.state[i] ^ (previous ^ (previous >> 30)).wrapping_mul(1_664_525))
                .wrapping_add(key[j])
                .wrapping_add(j as u32);
            i += 1;
            j += 1;
            if i >= 624 {
                self.state[0] = self.state[623];
                i = 1;
            }
            if j >= key.len() {
                j = 0;
            }
        }
        for _ in 0..623 {
            let previous = self.state[i - 1];
            self.state[i] =
                (self.state[i] ^ (previous ^ (previous >> 30)).wrapping_mul(1_566_083_941)).wrapping_sub(i as u32);
            i += 1;
            if i >= 624 {
                self.state[0] = self.state[623];
                i = 1;
            }
        }
        self.state[0] = 0x8000_0000;
    }

    fn generate(&mut self) {
        for i in 0..624 {
            let y = (self.state[i] & 0x8000_0000) | (self.state[(i + 1) % 624] & 0x7fff_ffff);
            let mut next = self.state[(i + 397) % 624] ^ (y >> 1);
            if y & 1 != 0 {
                next ^= 0x9908_b0df;
            }
            self.state[i] = next;
        }
        self.index = 0;
    }

    /// The next 32 random bits.
    pub fn next_u32(&mut self) -> u32 {
        if self.index >= 624 {
            self.generate();
        }
        let mut y = self.state[self.index];
        self.index += 1;
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c_5680;
        y ^= (y << 15) & 0xefc6_0000;
        y ^ (y >> 18)
    }

    /// `random.getrandbits(bits)` for `bits <= 32`.
    pub fn getrandbits(&mut self, bits: u32) -> u32 {
        debug_assert!((1..=32).contains(&bits));
        self.next_u32() >> (32 - bits)
    }

    /// `random._randbelow(n)`: a uniform integer below `n` by rejection sampling.
    pub fn randbelow(&mut self, n: u32) -> u32 {
        let bits = 32 - n.leading_zeros();
        loop {
            let candidate = self.getrandbits(bits);
            if candidate < n {
                return candidate;
            }
        }
    }

    /// `random.shuffle(items)`.
    pub fn shuffle<T>(&mut self, items: &mut [T]) {
        for i in (1..items.len()).rev() {
            let j = self.randbelow((i + 1) as u32) as usize;
            items.swap(i, j);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn draws_what_cpython_draws() {
        // random.seed(42); [random.getrandbits(32) for _ in range(3)]
        let mut random = PyRandom::new(42);
        let drawn: Vec<u32> = (0..3).map(|_| random.getrandbits(32)).collect();
        assert_eq!(drawn, [2_746_317_213, 478_163_327, 107_420_369]);
        // x = list(range(300)); random.seed(42); random.shuffle(x)
        let mut items: Vec<u32> = (0..300).collect();
        PyRandom::new(42).shuffle(&mut items);
        assert_eq!(items[..12], [106, 239, 37, 89, 32, 250, 192, 120, 19, 225, 24, 295]);
        assert_eq!(items[295..], [114, 125, 140, 12, 57]);
    }
}
