//! Is a return series a random walk? `cargo run --example variance_ratio`
mod common;
use quars::math::variance_ratio::variance_ratio;

fn main() -> Result<(), quars::Error> {
    for (name, phi) in [
        ("independent", 0.0),
        ("persistent", 0.5),
        ("mean-reverting", -0.5),
    ] {
        let r = variance_ratio(&common::ar1(phi, 1024), 4)?;
        println!(
            "{name:<15} VR(4) = {:.3}  z* = {:+.2}  p = {:.4}",
            r.vr, r.z_robust, r.p_robust
        );
    }
    Ok(())
}
