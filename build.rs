fn main() {
    emit_ring();

    // Build GIT_SHA env when snapshot profiling is enabled
    #[cfg(feature = "profile")]
    {
        let sha = std::process::Command::new("git")
            .args(["rev-parse", "--short", "HEAD"])
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .unwrap_or_else(|| "unknown".to_string());
        println!("cargo:rustc-env=GIT_SHA={sha}");
        println!("cargo:rerun-if-changed=.git/HEAD");
        println!("cargo:rerun-if-changed=.git/refs");
    }
}

/// The cyclotomic family of the ring spec's `family` field.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Family {
    /// `Z_q[X]/(X^N + 1)`, `N` a power of two (conductor `2N`): the default.
    PowerOfTwo,
    /// `Z_p[X]/(X^{2n} - X^n + 1)`, `n = 3 * 2^{k-1}`, of conductor `9 * 2^k` and degree
    /// `phi = 3 * 2^k`, over the Goldilocks prime `p = 2^64 - 2^32 + 1`.
    Trinomial,
}

impl Family {
    fn name(self) -> &'static str {
        match self {
            Family::PowerOfTwo => "power_of_two",
            Family::Trinomial => "trinomial",
        }
    }
}

struct RingSpec {
    family: Family,
    degree: u64,
    mod_q: u64,
    tau: u64,
    op_norm_bound: f64,
    projection_batches: u64,
}

fn emit_ring() {
    println!("cargo:rerun-if-env-changed=ROKOKO_RING");
    let manifest_dir = std::path::PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
    let path = match std::env::var_os("ROKOKO_RING") {
        Some(path) => manifest_dir.join(path),
        None => manifest_dir.join("rings/default.toml"),
    };
    println!("cargo:rerun-if-changed={}", path.display());
    println!("cargo::rustc-check-cfg=cfg(rokoko_ring, values(any()))");
    println!("cargo::rustc-check-cfg=cfg(rokoko_family, values(\"power_of_two\", \"trinomial\"))");
    if let Some(stem) = path.file_stem() {
        println!("cargo::rustc-cfg=rokoko_ring={:?}", stem.to_string_lossy());
    }
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|e| fail(&path, &format!("cannot read the ring spec: {e}")));
    let spec = parse_ring(&text).unwrap_or_else(|e| fail(&path, &e));
    if let Err(e) = validate_ring(&spec) {
        fail(&path, &e);
    }
    println!("cargo::rustc-cfg=rokoko_family={:?}", spec.family.name());
    let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap());
    let slot_degree = match spec.family {
        Family::PowerOfTwo => slot_degree(spec.degree, spec.mod_q),
        Family::Trinomial => 3,
    };
    let family = match spec.family {
        Family::PowerOfTwo => "PowerOfTwo",
        Family::Trinomial => "Trinomial",
    };
    // `HALF_DEGREE` is `N/2` for `X^N + 1` and `n = phi/2` for `X^{2n} - X^n + 1`.
    std::fs::write(
        out.join("ring.rs"),
        format!(
            "/// The cyclotomic family of the ring, from the ring spec's `family` field.\n\
             #[derive(Clone, Copy, PartialEq, Eq, Debug)]\n\
             pub enum RingFamily {{\n    /// `Z_q[X]/(X^N + 1)`, `N` a power of two.\n    PowerOfTwo,\n    \
             /// `Z_p[X]/(X^{{2n}} - X^n + 1)`, conductor `9 * 2^k`, over Goldilocks.\n    Trinomial,\n}}\n\
             pub const RING_FAMILY: RingFamily = RingFamily::{};\n\
             pub const DEGREE: usize = {};\npub const HALF_DEGREE: usize = {};\npub const MOD_Q: u64 = {};\n\
             pub const SLOT_DEGREE: usize = {};\npub const NUM_SLOTS: usize = {};\n\
             pub const NOF_BATCHES: usize = {};\n",
            family,
            spec.degree,
            spec.degree / 2,
            spec.mod_q,
            slot_degree,
            spec.degree / slot_degree,
            spec.projection_batches,
        ),
    )
    .unwrap();
    std::fs::write(
        out.join("challenge.rs"),
        format!(
            "pub const TAU: usize = {};\npub const T_OP_NORM_BOUND: f64 = {:?};\n",
            spec.tau, spec.op_norm_bound
        ),
    )
    .unwrap();
}

fn fail(path: &std::path::Path, message: &str) -> ! {
    println!("cargo::error=ring spec {}: {message}", path.display());
    std::process::exit(0);
}

fn parse_ring(text: &str) -> Result<RingSpec, String> {
    let mut fields = std::collections::HashMap::new();
    for line in text.lines() {
        let line = line.split('#').next().unwrap().trim();
        if line.is_empty() {
            continue;
        }
        let (key, value) = line
            .split_once('=')
            .ok_or_else(|| format!("expected `key = value`, found {line:?}"))?;
        let (key, value) = (key.trim(), value.trim().to_string());
        if fields.insert(key.to_string(), value).is_some() {
            return Err(format!("`{key}` is given twice"));
        }
    }
    // A string, quoted as in TOML.
    let family = fields.remove("family");
    let family = match family.as_deref().map(|value| value.trim_matches('"')) {
        None | Some("power_of_two") => Family::PowerOfTwo,
        Some("trinomial") => Family::Trinomial,
        Some(other) => {
            return Err(format!(
                "`family` must be \"power_of_two\" or \"trinomial\", found {other:?}"
            ))
        }
    };
    let mut take = |key: &str| fields.remove(key).ok_or_else(|| format!("missing `{key}`"));
    // Numbers may use `_` as a digit separator; strings such as `family` keep theirs.
    let int = |key: &str, value: String| {
        value
            .replace('_', "")
            .parse::<u64>()
            .map_err(|_| format!("`{key}` must be an unsigned integer, found {value:?}"))
    };
    let spec = RingSpec {
        family,
        degree: int("degree", take("degree")?)?,
        mod_q: int("mod_q", take("mod_q")?)?,
        tau: int("tau", take("tau")?)?,
        op_norm_bound: {
            let value = take("op_norm_bound")?;
            value
                .replace('_', "")
                .parse::<f64>()
                .map_err(|_| format!("`op_norm_bound` must be a number, found {value:?}"))?
        },
        projection_batches: match fields.remove("projection_batches") {
            Some(value) => int("projection_batches", value)?,
            None => 2,
        },
    };
    match fields.keys().next() {
        Some(key) => Err(format!("unknown field `{key}`")),
        None => Ok(spec),
    }
}

const MIN_DEGREE: u64 = 128;
const MAX_DEGREE: u64 = 256;
const Q_MIN_BITS: u32 = 15;
const Q_MAX_BITS: u32 = 50;

/// The Goldilocks prime `2^64 - 2^32 + 1`, the modulus of the trinomial family.
const GOLDILOCKS_P: u64 = 0xffff_ffff_0000_0001;
/// The exponents `k` of the trinomial family's degrees `3 * 2^k`.
const TRINOMIAL_K: [u32; 2] = [5, 6];

fn validate_ring(spec: &RingSpec) -> Result<(), String> {
    match spec.family {
        Family::PowerOfTwo => validate_power_of_two(spec)?,
        Family::Trinomial => validate_trinomial(spec)?,
    }
    let n = spec.degree;
    if spec.tau == 0 || spec.tau > n {
        return Err(format!("tau {} must lie in [1, degree]", spec.tau));
    }
    if !(spec.op_norm_bound.is_finite() && spec.op_norm_bound > 0.0) {
        return Err(format!(
            "op_norm_bound {} must be positive",
            spec.op_norm_bound
        ));
    }
    if !matches!(spec.projection_batches, 2 | 3) {
        return Err(format!(
            "projection_batches {} must be 2 or 3",
            spec.projection_batches
        ));
    }
    Ok(())
}

/// `X^{2n} - X^n + 1 = Phi_{9 * 2^k}` of degree `3 * 2^k` over Goldilocks: it splits into `2^k`
/// irreducible cubics `X^3 - eta_j`, `eta_j` the primitive `3 * 2^k`-th roots of unity.
fn validate_trinomial(spec: &RingSpec) -> Result<(), String> {
    let (n, q) = (spec.degree, spec.mod_q);
    if !TRINOMIAL_K.iter().any(|&k| n == 3 << k) {
        return Err(format!(
            "degree {n} must be 3 * 2^k with k in {TRINOMIAL_K:?} for the trinomial family"
        ));
    }
    if q != GOLDILOCKS_P {
        return Err(format!(
            "mod_q {q} must be the Goldilocks prime {GOLDILOCKS_P} for the trinomial family"
        ));
    }
    Ok(())
}

fn validate_power_of_two(spec: &RingSpec) -> Result<(), String> {
    let (n, q) = (spec.degree, spec.mod_q);
    if !n.is_power_of_two() || !(MIN_DEGREE..=MAX_DEGREE).contains(&n) {
        return Err(format!(
            "degree {n} must be a power of two in [{MIN_DEGREE}, {MAX_DEGREE}]: the challenge \
             sampler draws indices as bytes"
        ));
    }
    if !(1 << Q_MIN_BITS < q && q < 1 << Q_MAX_BITS) {
        return Err(format!(
            "mod_q {q} must lie in (2^{Q_MIN_BITS}, 2^{Q_MAX_BITS}): i16 digits are lifted by \
             adding q, and the IFMA NTT, the float-Barrett kernels and the lazy reduction budget \
             floor(2^64 / 4q) assume q < 2^{Q_MAX_BITS}"
        ));
    }
    if !is_prime(q) {
        return Err(format!("mod_q {q} is not prime"));
    }
    if !matches!(slot_degree(n, q), 2 | 4) {
        return Err(format!(
            "X^{n}+1 must split into irreducible factors of degree 2 or 4 mod q, so q - 1 must \
             have 2-adic valuation log2(degree) = {} or log2(degree) - 1 = {}; {q} - 1 has {}, \
             which gives factors of degree {}",
            n.trailing_zeros(),
            n.trailing_zeros() - 1,
            (q - 1).trailing_zeros(),
            slot_degree(n, q)
        ));
    }
    Ok(())
}

/// Degree `2N / 2^v2(q-1)` of the irreducible factors `X^d - zeta` of `X^N + 1` over `Z_q`, or 1
/// when `X^N + 1` splits completely.
fn slot_degree(n: u64, q: u64) -> u64 {
    (2 * n)
        .checked_shr((q - 1).trailing_zeros())
        .unwrap_or(0)
        .max(1)
}

fn is_prime(n: u64) -> bool {
    if n < 2 {
        return false;
    }
    let mul = |a: u64, b: u64| ((a as u128 * b as u128) % n as u128) as u64;
    let pow = |mut a: u64, mut e: u64| {
        let mut r = 1;
        while e > 0 {
            if e & 1 == 1 {
                r = mul(r, a);
            }
            a = mul(a, a);
            e >>= 1;
        }
        r
    };
    let witnesses = [2u64, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
    if let Some(&p) = witnesses.iter().find(|&&p| n % p == 0) {
        return n == p;
    }
    let s = (n - 1).trailing_zeros();
    let d = (n - 1) >> s;
    witnesses.iter().all(|&a| {
        let mut x = pow(a, d);
        if x == 1 || x == n - 1 {
            return true;
        }
        (1..s).any(|_| {
            x = mul(x, x);
            x == n - 1
        })
    })
}
