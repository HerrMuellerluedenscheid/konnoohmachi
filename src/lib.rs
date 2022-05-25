use ndarray::{self, Array, Array1, ArrayBase, OwnedRepr, Zip};
use numpy::{IntoPyArray, PyArray1, PyArrayDyn};
use pyo3::prelude::{pymodule, PyModule, PyResult, Python};

#[pymodule]
fn konnoohmachi(_py: Python<'_>, m: &PyModule) -> PyResult<()> {

    #[pyfn(m)]
    fn smooth<'py>(
        py: Python<'py>,
        frequencies: &PyArrayDyn<f64>,
        amplitudes: &PyArrayDyn<f64>,
        bandwidth: f64,
    ) -> &'py PyArray1<f64> {
        let frequencies = unsafe { frequencies.as_array() };
        let amplitudes = unsafe { amplitudes.as_array() };

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
        out.into_pyarray(py)
    }

    Ok(())
}
