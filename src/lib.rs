use ndarray::{self, Array, Array1, ArrayBase, Dim, IxDynImpl, OwnedRepr, ViewRepr, Zip};
use numpy::{IntoPyArray, PyArray1, PyArrayDyn};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

/// Python module that implements konnoohmachi spectral smoothing
#[pymodule]
fn konnoohmachi(_py: Python<'_>, m: &PyModule) -> PyResult<()> {
    /// Python interface: Smooth a spectrum provided as two one-dimensional vectors containing
    /// `frequencies` and `amplitudes`
    #[pyfn(m)]
    fn smooth<'py>(
        py: Python<'py>,
        frequencies: &PyArrayDyn<f64>,
        amplitudes: &PyArrayDyn<f64>,
        bandwidth: f64,
    ) -> &'py PyArray1<f64> {
        let frequencies = unsafe { frequencies.as_array() };
        let amplitudes = unsafe { amplitudes.as_array() };

        let out = konnoohmachi_smooth(frequencies, amplitudes, bandwidth);
        out.into_pyarray(py)
    }

    Ok(())
}

/// Smooth a spectrum provided as two one-dimensional vectors containing
/// `frequencies` and `amplitudes`
pub fn konnoohmachi_smooth(
    frequencies: ArrayBase<ViewRepr<&f64>, Dim<IxDynImpl>>,
    amplitudes: ArrayBase<ViewRepr<&f64>, Dim<IxDynImpl>>,
    bandwidth: f64,
) -> ArrayBase<OwnedRepr<f64>, Dim<[usize; 1]>> {
    assert_eq!(
        frequencies.len(),
        amplitudes.len(),
        "amplitudes and frequencies have to have equal length."
    );

    assert!(bandwidth > 0.0, "bandwidth has to be greater than 0.");

    let mut out = Array1::<f64>::ones(frequencies.len());

    let zero_frequency_index = frequencies.iter().position(|&v| v == 0.0).unwrap();

    let mut frequencies_work: ArrayBase<OwnedRepr<f64>, _> = Array::zeros(frequencies.raw_dim());
    frequencies_work.assign(&frequencies);

    for (index_frequency, corner_frequency) in frequencies.iter().enumerate() {
        Zip::from(&mut frequencies_work)
            .and(&frequencies)
            .for_each(|freq_work, &freq| {
                *freq_work = f64::log10(freq / corner_frequency * bandwidth)
            });

        frequencies_work.map_inplace(|w| *w = f64::powi(f64::sin(*w) / *w, 4));
        frequencies_work[index_frequency] = 1.;
        frequencies_work[zero_frequency_index] = 0.;

        let normalization = frequencies_work.sum();

        Zip::from(&mut frequencies_work)
            .and(&amplitudes)
            .for_each(|w, &a| *w = (*w * a) / normalization);

        out[index_frequency] = frequencies_work.sum();
    }
    out[zero_frequency_index] = 0.;
    out
}

#[cfg(test)]
mod tests {
    use ndarray::Array1;

    use crate::konnoohmachi_smooth;

    #[test]
    fn test_basic_ndarray() {
        let frequencies = Array1::<f64>::zeros(10);
        let amplitudes = Array1::<f64>::ones(10);
        let bandwidth = 1.0;
        konnoohmachi_smooth(
            frequencies.view().into_dyn(),
            amplitudes.view().into_dyn(),
            bandwidth,
        );
    }
}
