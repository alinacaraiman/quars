use crate::Error;

/// Fixed-rate bond repaid at maturity, valued on a coupon date (no accrued interest)
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Bond {
    pub face: f64,
    /// Annual coupon rate, 0 for a zero-coupon bond
    pub coupon_rate: f64,
    /// Coupon periods per year
    pub frequency: u32,
    /// Coupon periods left until maturity
    pub periods: u32,
}

impl Bond {
    /// Present value of each payment as (period, value) at an annual `yield_rate`
    /// compounded `frequency` times a year
    fn present_values(&self, yield_rate: f64) -> impl Iterator<Item = (f64, f64)> + '_ {
        let frequency = self.frequency as f64;
        let coupon = self.face * self.coupon_rate / frequency;
        (1..=self.periods).map(move |t| {
            let payment = if t == self.periods {
                coupon + self.face
            } else {
                coupon
            };
            (
                t as f64,
                payment / (1.0 + yield_rate / frequency).powi(t as i32),
            )
        })
    }

    pub fn price(&self, yield_rate: f64) -> f64 {
        self.present_values(yield_rate).map(|(_, pv)| pv).sum()
    }

    /// Present-value weighted time to the payments, in years
    pub fn macaulay_duration(&self, yield_rate: f64) -> f64 {
        let weighted: f64 = self.present_values(yield_rate).map(|(t, pv)| t * pv).sum();
        weighted / self.price(yield_rate) / self.frequency as f64
    }

    /// Relative price change per unit change in yield, −(dP/dy)/P
    pub fn modified_duration(&self, yield_rate: f64) -> f64 {
        self.macaulay_duration(yield_rate) / (1.0 + yield_rate / self.frequency as f64)
    }

    /// (d²P/dy²)/P
    pub fn convexity(&self, yield_rate: f64) -> f64 {
        let frequency = self.frequency as f64;
        let weighted: f64 = self
            .present_values(yield_rate)
            .map(|(t, pv)| t * (t + 1.0) * pv)
            .sum();
        weighted / (self.price(yield_rate) * (frequency + yield_rate).powi(2))
    }

    /// Yield at which the bond is worth `price`, by bisection
    pub fn yield_to_maturity(&self, price: f64) -> Result<f64, Error> {
        let (mut low, mut high) = (-0.5, 10.0);
        if !(price < self.price(low) && price > self.price(high)) {
            return Err("Price implies a yield outside -50% to 1000%.".into());
        }
        for _ in 0..100 {
            let mid = 0.5 * (low + high);
            if self.price(mid) > price {
                low = mid;
            } else {
                high = mid;
            }
        }
        Ok(0.5 * (low + high))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(actual: f64, expected: f64) {
        assert!((actual - expected).abs() < 1e-8, "{actual} != {expected}");
    }

    // Expected values from the textbook sums evaluated with numpy
    #[test]
    fn coupon_bond() {
        let bond = Bond {
            face: 1000.0,
            coupon_rate: 0.06,
            frequency: 2,
            periods: 10,
        };
        assert_close(bond.price(0.04), 1089.825850062);
        assert_close(bond.macaulay_duration(0.04), 4.423465253);
        assert_close(bond.modified_duration(0.04), 4.336730641);
        assert_close(bond.convexity(0.04), 22.394877960);
        assert_close(bond.yield_to_maturity(950.0).unwrap(), 0.072087477642);
    }

    #[test]
    fn par_and_zero_coupon() {
        let par = Bond {
            face: 100.0,
            coupon_rate: 0.05,
            frequency: 1,
            periods: 10,
        };
        assert_close(par.price(0.05), 100.0);
        assert_close(par.macaulay_duration(0.05), 8.107821676);
        let zero = Bond {
            face: 100.0,
            coupon_rate: 0.0,
            frequency: 2,
            periods: 6,
        };
        assert_close(zero.price(0.05), 86.229686596);
        assert_close(zero.macaulay_duration(0.05), 3.0);
        assert_close(zero.convexity(0.05), 9.994051160);
    }

    #[test]
    fn duration_and_convexity_are_price_derivatives() {
        let bond = Bond {
            face: 100.0,
            coupon_rate: 0.03,
            frequency: 2,
            periods: 14,
        };
        let (y, h) = (0.045, 1e-5);
        let (down, mid, up) = (bond.price(y - h), bond.price(y), bond.price(y + h));
        assert!((bond.modified_duration(y) + (up - down) / (2.0 * h) / mid).abs() < 1e-6);
        assert!((bond.convexity(y) - (up - 2.0 * mid + down) / (h * h) / mid).abs() < 1e-3);
    }

    #[test]
    fn rejects_unreachable_price() {
        let bond = Bond {
            face: 100.0,
            coupon_rate: 0.05,
            frequency: 1,
            periods: 10,
        };
        assert!(bond.yield_to_maturity(0.0).is_err());
    }
}
