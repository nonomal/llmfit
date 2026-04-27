//! Client for the localmaxxing.com benchmark API.
//!
//! Fetches real-world benchmark results (tok/s, TTFT, VRAM usage) for
//! hardware configurations that match the user's detected system specs.

use crate::hardware::{GpuBackend, SystemSpecs};
use serde::{Deserialize, Serialize};

const BASE_URL: &str = "https://localmaxxing.com/api";

// ── Response types ───────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BenchmarkEntry {
    pub id: String,
    #[serde(default)]
    pub hf_id: String,
    #[serde(default)]
    pub engine_name: String,
    #[serde(default)]
    pub quantization: String,
    #[serde(default)]
    pub tok_s_out: Option<f64>,
    #[serde(default)]
    pub tok_s_total: Option<f64>,
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub peak_vram_gb: Option<f64>,
    #[serde(default)]
    pub notes: Option<String>,
    #[serde(default)]
    pub hardware: Option<HardwareInfo>,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub verified: Option<bool>,
    #[serde(default)]
    pub created_at: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct HardwareInfo {
    #[serde(default)]
    pub hw_class: Option<String>,
    #[serde(default)]
    pub gpu_name: Option<String>,
    #[serde(default)]
    pub vram_gb: Option<f64>,
    #[serde(default)]
    pub gpu_count: Option<u32>,
    #[serde(default)]
    pub chip_vendor: Option<String>,
    #[serde(default)]
    pub chip_family: Option<String>,
    #[serde(default)]
    pub chip_variant: Option<String>,
    #[serde(default)]
    pub unified_memory_gb: Option<f64>,
    #[serde(default)]
    pub cpu: Option<String>,
    #[serde(default)]
    pub ram_gb: Option<f64>,
    #[serde(default)]
    pub os: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct LeaderboardEntry {
    pub id: String,
    #[serde(default)]
    pub hf_id: String,
    #[serde(default)]
    pub engine_name: String,
    #[serde(default)]
    pub quantization: String,
    #[serde(default)]
    pub tok_s_out: Option<f64>,
    #[serde(default)]
    pub tok_s_total: Option<f64>,
    #[serde(default)]
    pub ttft_ms: Option<f64>,
    #[serde(default)]
    pub context_length: Option<u32>,
    #[serde(default)]
    pub batch_size: Option<u32>,
    #[serde(default)]
    pub peak_vram_gb: Option<f64>,
    #[serde(default)]
    pub hardware_name: Option<String>,
    #[serde(default)]
    pub username: Option<String>,
    #[serde(default)]
    pub verified: Option<bool>,
    #[serde(default)]
    pub param_size: Option<String>,
    #[serde(default)]
    pub model_family: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BenchmarkResponse {
    pub benchmarks: Vec<BenchmarkEntry>,
    #[serde(default)]
    pub total: u64,
    #[serde(default)]
    pub limit: u64,
    #[serde(default)]
    pub offset: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LeaderboardResponse {
    pub rows: Vec<LeaderboardEntry>,
    #[serde(default)]
    pub total: u64,
    #[serde(default)]
    pub limit: u64,
    #[serde(default)]
    pub offset: u64,
}

// ── Query builder ────────────────────────────────────────────────────

/// Map detected hardware to API query parameters for matching benchmarks.
pub fn hw_query_params(specs: &SystemSpecs) -> Vec<(&'static str, String)> {
    let mut params: Vec<(&str, String)> = Vec::new();

    if specs.unified_memory {
        params.push(("hwClass", "UNIFIED".to_string()));

        // Apple Silicon
        if specs.backend == GpuBackend::Metal {
            params.push(("chipVendor", "apple".to_string()));
            if let Some(ref gpu) = specs.gpu_name {
                // e.g. "Apple M2 Max" → chipFamily "m2", chipVariant "max"
                let lower = gpu.to_lowercase();
                if let Some(rest) = lower.strip_prefix("apple ") {
                    let parts: Vec<&str> = rest.split_whitespace().collect();
                    if !parts.is_empty() {
                        params.push(("chipFamily", parts[0].to_string()));
                    }
                    if parts.len() > 1 {
                        params.push(("chipVariant", parts[1].to_string()));
                    }
                }
            }
        }
    } else if specs.has_gpu {
        params.push(("hwClass", "DISCRETE_GPU".to_string()));

        if let Some(ref name) = specs.gpu_name {
            params.push(("gpuName", name.clone()));
        }
    } else {
        params.push(("hwClass", "CPU_ONLY".to_string()));
    }

    params
}

/// Map detected hardware to leaderboard query parameters.
pub fn hw_leaderboard_params(specs: &SystemSpecs) -> Vec<(&'static str, String)> {
    let mut params: Vec<(&str, String)> = Vec::new();

    if specs.unified_memory {
        params.push(("hwClass", "UNIFIED".to_string()));
    } else if specs.has_gpu {
        params.push(("hwClass", "DISCRETE_GPU".to_string()));
    } else {
        params.push(("hwClass", "CPU_ONLY".to_string()));
    }

    // Use hardware name for fuzzy match
    if let Some(ref name) = specs.gpu_name {
        params.push(("hardwareName", name.clone()));
    }

    // VRAM tier
    if let Some(vram) = specs.total_gpu_vram_gb {
        let tier = nearest_mem_tier(vram);
        if tier > 0 {
            params.push(("memTier", tier.to_string()));
        }
    } else if specs.unified_memory {
        let tier = nearest_mem_tier(specs.total_ram_gb);
        if tier > 0 {
            params.push(("memTier", tier.to_string()));
        }
    }

    // OS
    let os = if cfg!(target_os = "macos") {
        "macos"
    } else if cfg!(target_os = "windows") {
        "windows"
    } else {
        "linux"
    };
    params.push(("os", os.to_string()));

    params
}

fn nearest_mem_tier(gb: f64) -> u32 {
    const TIERS: [u32; 9] = [8, 12, 16, 24, 32, 48, 80, 96, 128];
    let mut best = 0u32;
    let mut best_dist = f64::MAX;
    for &t in &TIERS {
        let d = (gb - t as f64).abs();
        if d < best_dist {
            best_dist = d;
            best = t;
        }
    }
    best
}

// ── Fetch functions ──────────────────────────────────────────────────

/// Fetch benchmarks matching the user's hardware.
pub fn fetch_benchmarks(
    specs: &SystemSpecs,
    api_key: Option<&str>,
    limit: u32,
) -> Result<BenchmarkResponse, String> {
    let mut params = hw_query_params(specs);
    params.push(("limit", limit.to_string()));

    let query: String = params
        .iter()
        .map(|(k, v)| format!("{}={}", k, urlencoded(v)))
        .collect::<Vec<_>>()
        .join("&");

    let url = format!("{}/benchmarks?{}", BASE_URL, query);
    let mut req = ureq::get(&url);
    if let Some(key) = api_key {
        req = req.header("Authorization", &format!("Bearer {}", key));
    }
    let resp = req.call().map_err(|e| format!("HTTP error: {}", e))?;
    let body: BenchmarkResponse = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON parse error: {}", e))?;
    Ok(body)
}

/// Fetch benchmarks for a specific model on matching hardware.
pub fn fetch_benchmarks_for_model(
    specs: &SystemSpecs,
    hf_id: &str,
    api_key: Option<&str>,
    limit: u32,
) -> Result<BenchmarkResponse, String> {
    let mut params = hw_query_params(specs);
    params.push(("hfId", hf_id.to_string()));
    params.push(("limit", limit.to_string()));

    let query: String = params
        .iter()
        .map(|(k, v)| format!("{}={}", k, urlencoded(v)))
        .collect::<Vec<_>>()
        .join("&");

    let url = format!("{}/benchmarks?{}", BASE_URL, query);
    let mut req = ureq::get(&url);
    if let Some(key) = api_key {
        req = req.header("Authorization", &format!("Bearer {}", key));
    }
    let resp = req.call().map_err(|e| format!("HTTP error: {}", e))?;
    let body: BenchmarkResponse = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON parse error: {}", e))?;
    Ok(body)
}

/// Fetch the leaderboard filtered to matching hardware.
pub fn fetch_leaderboard(
    specs: &SystemSpecs,
    api_key: Option<&str>,
    limit: u32,
) -> Result<LeaderboardResponse, String> {
    let mut params = hw_leaderboard_params(specs);
    params.push(("limit", limit.to_string()));

    let query: String = params
        .iter()
        .map(|(k, v)| format!("{}={}", k, urlencoded(v)))
        .collect::<Vec<_>>()
        .join("&");

    let url = format!("{}/leaderboard?{}", BASE_URL, query);
    let mut req = ureq::get(&url);
    if let Some(key) = api_key {
        req = req.header("Authorization", &format!("Bearer {}", key));
    }
    let resp = req.call().map_err(|e| format!("HTTP error: {}", e))?;
    let body: LeaderboardResponse = resp
        .into_body()
        .read_json()
        .map_err(|e| format!("JSON parse error: {}", e))?;
    Ok(body)
}

/// Minimal percent-encoding for query values.
fn urlencoded(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for b in s.bytes() {
        match b {
            b'A'..=b'Z' | b'a'..=b'z' | b'0'..=b'9' | b'-' | b'_' | b'.' | b'~' => {
                out.push(b as char)
            }
            b' ' => out.push('+'),
            _ => {
                out.push('%');
                out.push(char::from(b"0123456789ABCDEF"[(b >> 4) as usize]));
                out.push(char::from(b"0123456789ABCDEF"[(b & 0xf) as usize]));
            }
        }
    }
    out
}
