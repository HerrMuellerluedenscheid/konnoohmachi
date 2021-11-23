#![allow(non_snake_case)]
#![allow(unused_imports)]

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

use numpy::ndarray::{ArrayD, ArrayViewD, ArrayViewMutD};
use numpy::{c64, IntoPyArray, PyArrayDyn, PyReadonlyArrayDyn};

// A base line implementation for konnoohmachi spectral filter with a python
// interface.

fn smoothing_window(freqs: ArrayViewMutD<'_, f64>, f_corner: f64, b: f64) {
    // Note that there HAS to be a zero frequency at the moment!
    let error_margin = 0.000000001;

    let index_zero_freq = freqs.iter().position(|&f| f == 0.0).unwrap();
    let index_f_corner = freqs.iter().position(|&f| (f - f_corner).abs() < error_margin).unwrap();

    // return freqs just for testing
    // &freqs  * b
    // if f_corner == 0.0 {
    //     let mut window = vec![0.; freqs.len()];
    //     window[index_zero_freq] = 1.;
    //     window
    // } else {
    println!("f_corner: {} b: {}", f_corner, b);
    // let scaled = (&freqs / f_corner) * b;
    // for (ival, val) in scaled.iter().enumerate() {
    //     println!("scaled: {} unscaled {}", val, &freqs[ival]);
    // }
    &freqs.iter()
        .map(|freq| f64::log10(freq / f_corner) * b)
        .map(|w| f64::powi(f64::sin(w) / w, 4));
    // freqs[index_f_corner] = 1.;
    // freqs[index_zero_freq] = 0.;
    // let normalization: f64 = freqs.iter().sum();
    // freqs.iter_mut().for_each(|x| *x /= normalization);
    
}

pub fn konnoohmachi_smooth(freqs: ArrayViewD<'_, f64>, amps: ArrayViewD<'_, f64>, b: f64) ->  ArrayD<f64> {
    // return freqs for testing
    b * &freqs
    // let n_freqs = freqs.len();
    // let mut smoothed = vec![0.; n_freqs];

    // let freqs_iter = freqs.clone();
    // for (i_freq, f_corner) in freqs_iter.iter().enumerate() {
    //     let window = smoothing_window(&freqs, *f_corner, b);

    //     let product: f64 = window.iter().zip(amps.iter()).map(|(x, y)| x * y).sum();

    //     smoothed[i_freq] = product;
    // }
    // smoothed
}

#[pymodule]
fn konnoohmachi(_py: Python, m: &PyModule) -> PyResult<()> {

    #[pyfn(m)] #[pyo3(name = "smooth")]
    fn smooth_py<'py>(py: Python<'py>, freqs: PyReadonlyArrayDyn<'_, f64>, amps: PyReadonlyArrayDyn<'_, f64>, b: f64) -> &'py PyArrayDyn<f64> {
        let freqs = freqs.as_array();
        let amps = amps.as_array();
        let smoothed = konnoohmachi_smooth(freqs, amps, b);
        smoothed.into_pyarray(py)
    }

    #[pyfn(m)] #[pyo3(name = "window")]
    fn window_py<'py>(py: Python<'py>, freqs: PyReadonlyArrayDyn<'_, f64>, f_corner: f64, b: f64) {
        let freqs = unsafe { freqs.as_array_mut() };
        smoothing_window(freqs, f_corner, b);
        // freqs.into_pyarray(py)
        // freqs
    }

    #[pyfn(m)] #[pyo3(name = "__version__")]
    fn version_py() -> PyResult<String> {
        Ok("0.1.4".to_string())
    }

    Ok(())
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_konnoohmachi_zero() {
        let amps = vec![0.0; 5];
        let freqs = vec![0.0; 5];
        let b = 0.0;

        assert_eq!(konnoohmachi_smooth(freqs, amps.clone(), b), amps);
    }

    #[test]
    fn test_konnoohmachi() {
        let amps = vec![3., 1., 3., 4.];
        let freqs = vec![0., 1., 2., 3.];
        let b = 1.0;

        assert_eq!(konnoohmachi_smooth(freqs, amps.clone(), b), [3.0, 2.5921214954009724, 2.6908685219248403, 2.7475508921225154]);
    }

    #[test]
    fn test_smoothing_window_zero() {
        let freqs = vec![0.0, 1.0, f64::log10(2.0)];
        let out_expect = vec![1., 0., 0.];
        let f_corner = 0.0;
        let b = 0.0;

        assert_eq!(smoothing_window(&freqs, f_corner, b), out_expect);
    }

    // #[test]
    // fn test_smoothing_window() {
    //     let freqs = vec![0.0, 1.0, 2.0];
    //     let mut result = f64::log10(2.0);
    //     result = f64::powi(result.sin() / result, 4);
    //     let out_expect = vec![0.0, 1.0, result];
    //     let f_corner = 1.0;
    //     let b = 1.0;
    //
    //     // assert_eq!(smoothing_window(freqs.clone(), f_corner, b), out_expect);
    // }
}
