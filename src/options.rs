use crate::math::{normal_cdf, normal_pdf};
use crate::Error;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum OptionKind {
    Call,
    Put,
}

/// Sensitivities of a Black (1976) price
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Greeks {
    /// To the forward or futures price
    pub delta: f64,
    pub gamma: f64,
    /// To volatility, per 1.00 (100 vol points)
    pub vega: f64,
    /// To the passage of time, per year
    pub theta: f64,
    /// To the interest rate, per 1.00
    pub rho: f64,
}

// Discount factor, d1 and d2
fn terms(forward: f64, strike: f64, rate: f64, volatility: f64, expiry: f64) -> (f64, f64, f64) {
    let total = volatility * expiry.sqrt();
    let d1 = ((forward / strike).ln() + 0.5 * total * total) / total;
    ((-rate * expiry).exp(), d1, d1 - total)
}

/// Black (1976) price of a European option on a forward or futures price.
/// `rate` and `volatility` are annual, `expiry` is in years.
pub fn black76(
    kind: OptionKind,
    forward: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    expiry: f64,
) -> f64 {
    let sign = if kind == OptionKind::Call { 1.0 } else { -1.0 };
    let discount = (-rate * expiry).exp();
    if !(volatility * expiry.sqrt() > 0.0) {
        return discount * (sign * (forward - strike)).max(0.0);
    }
    let (_, d1, d2) = terms(forward, strike, rate, volatility, expiry);
    discount * sign * (forward * normal_cdf(sign * d1) - strike * normal_cdf(sign * d2))
}

/// Black-Scholes-Merton price of a European option on a spot price paying a continuous
/// `dividend_yield`: Black (1976) on the forward spot · e^((rate − dividend_yield) · expiry)
pub fn black_scholes(
    kind: OptionKind,
    spot: f64,
    strike: f64,
    rate: f64,
    dividend_yield: f64,
    volatility: f64,
    expiry: f64,
) -> f64 {
    let forward = spot * ((rate - dividend_yield) * expiry).exp();
    black76(kind, forward, strike, rate, volatility, expiry)
}

pub fn black76_greeks(
    kind: OptionKind,
    forward: f64,
    strike: f64,
    rate: f64,
    volatility: f64,
    expiry: f64,
) -> Greeks {
    let (discount, d1, _) = terms(forward, strike, rate, volatility, expiry);
    let price = black76(kind, forward, strike, rate, volatility, expiry);
    let density = discount * normal_pdf(d1);
    Greeks {
        delta: match kind {
            OptionKind::Call => discount * normal_cdf(d1),
            OptionKind::Put => -discount * normal_cdf(-d1),
        },
        gamma: density / (forward * volatility * expiry.sqrt()),
        vega: forward * density * expiry.sqrt(),
        theta: -forward * density * volatility / (2.0 * expiry.sqrt()) + rate * price,
        rho: -expiry * price,
    }
}

/// Volatility at which `black76` gives `price`, by bisection
pub fn implied_volatility(
    kind: OptionKind,
    price: f64,
    forward: f64,
    strike: f64,
    rate: f64,
    expiry: f64,
) -> Result<f64, Error> {
    let value = |volatility| black76(kind, forward, strike, rate, volatility, expiry);
    let (mut low, mut high) = (0.0, 10.0);
    if !(price > value(low) && price < value(high)) {
        return Err("Price is outside the range of volatilities 0 to 1000%.".into());
    }
    for _ in 0..100 {
        let mid = 0.5 * (low + high);
        if value(mid) < price {
            low = mid;
        } else {
            high = mid;
        }
    }
    Ok(0.5 * (low + high))
}

#[cfg(test)]
mod tests {
    use super::OptionKind::{Call, Put};
    use super::*;

    fn assert_close(actual: f64, expected: f64, tolerance: f64) {
        assert!(
            (actual - expected).abs() < tolerance,
            "{actual} != {expected}"
        );
    }

    // Hull's textbook examples, digits from scipy
    #[test]
    fn prices_match_reference() {
        assert_close(
            black76(Put, 20.0, 20.0, 0.09, 0.25, 4.0 / 12.0),
            1.116641457,
            1e-8,
        );
        assert_close(
            black76(Call, 105.0, 100.0, 0.03, 0.2, 0.75),
            9.632639453,
            1e-8,
        );
        assert_close(
            black76(Put, 105.0, 100.0, 0.03, 0.2, 0.75),
            4.743883268,
            1e-8,
        );
        assert_close(
            black_scholes(Call, 42.0, 40.0, 0.1, 0.0, 0.2, 0.5),
            4.759422393,
            1e-8,
        );
        assert_close(
            black_scholes(Put, 42.0, 40.0, 0.1, 0.0, 0.2, 0.5),
            0.808599373,
            1e-8,
        );
    }

    #[test]
    fn put_call_parity_and_expiry() {
        let (f, k, r, t) = (105.0, 100.0, 0.03, 0.75);
        let parity = black76(Call, f, k, r, 0.2, t) - black76(Put, f, k, r, 0.2, t);
        assert_close(parity, (-r * t).exp() * (f - k), 1e-10);
        assert_close(black76(Call, f, k, r, 0.2, 0.0), 5.0, 1e-12);
        assert_close(black76(Put, f, k, r, 0.2, 0.0), 0.0, 1e-12);
    }

    #[test]
    fn greeks_are_price_derivatives() {
        let (f, k, r, v, t, h) = (105.0, 100.0, 0.03, 0.2, 0.75, 1e-4);
        for kind in [Call, Put] {
            let price = |f, r, v, t| black76(kind, f, k, r, v, t);
            let greeks = black76_greeks(kind, f, k, r, v, t);
            let slope = |up: f64, down: f64| (up - down) / (2.0 * h);
            assert_close(
                greeks.delta,
                slope(price(f + h, r, v, t), price(f - h, r, v, t)),
                1e-6,
            );
            assert_close(
                greeks.gamma,
                (price(f + h, r, v, t) - 2.0 * price(f, r, v, t) + price(f - h, r, v, t)) / (h * h),
                1e-4,
            );
            assert_close(
                greeks.vega,
                slope(price(f, r, v + h, t), price(f, r, v - h, t)),
                1e-6,
            );
            assert_close(
                greeks.theta,
                -slope(price(f, r, v, t + h), price(f, r, v, t - h)),
                1e-6,
            );
            assert_close(
                greeks.rho,
                slope(price(f, r + h, v, t), price(f, r - h, v, t)),
                1e-6,
            );
        }
    }

    #[test]
    fn implied_volatility_round_trip() {
        for kind in [Call, Put] {
            let price = black76(kind, 105.0, 100.0, 0.03, 0.2, 0.75);
            assert_close(
                implied_volatility(kind, price, 105.0, 100.0, 0.03, 0.75).unwrap(),
                0.2,
                1e-10,
            );
        }
        assert!(implied_volatility(Call, 0.0, 105.0, 100.0, 0.03, 0.75).is_err());
        assert!(implied_volatility(Call, 200.0, 105.0, 100.0, 0.03, 0.75).is_err());
    }
}
