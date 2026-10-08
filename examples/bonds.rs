//! Price, yield and risk of a coupon bond. `cargo run --example bonds`
use quars::bonds::Bond;

fn main() -> Result<(), quars::Error> {
    // Five years of semi-annual 6% coupons on a face value of 1000
    let bond = Bond {
        face: 1000.0,
        coupon_rate: 0.06,
        frequency: 2,
        periods: 10,
    };
    let price = bond.price(0.04);
    println!("price at a 4% yield  {price:.2}");
    println!("yield at that price  {:.4}", bond.yield_to_maturity(price)?);
    println!(
        "Macaulay duration    {:.3} years",
        bond.macaulay_duration(0.04)
    );
    println!("modified duration    {:.3}", bond.modified_duration(0.04));
    println!("convexity            {:.3}", bond.convexity(0.04));
    Ok(())
}
