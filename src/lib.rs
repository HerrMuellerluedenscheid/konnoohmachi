use ndarray::{
    Array1, Array2, ArrayBase, ArrayView1, ArrayViewMut1, Dim, Ix1, IxDynImpl, OwnedRepr, ViewRepr,
    Zip,
};
use numpy::{IntoPyArray, PyArray1, PyReadonlyArrayDyn};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Sentinel used when the frequency vector contains no zero frequency.
const NO_ZERO_FREQUENCY: usize = usize::MAX;

/// Python module that implements konnoohmachi spectral smoothing
#[pymodule]
fn konnoohmachi(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(smooth, m)?)?;
    m.add_class::<Smoother>()?;

    Ok(())
}

/// Python interface: Smooth a spectrum provided as two one-dimensional vectors containing
/// `frequencies` and `amplitudes`
#[pyfunction]
fn smooth<'py>(
    py: Python<'py>,
    frequencies: PyReadonlyArrayDyn<'py, f64>,
    amplitudes: PyReadonlyArrayDyn<'py, f64>,
    bandwidth: f64,
) -> Bound<'py, PyArray1<f64>> {
    let frequencies = frequencies.as_array();
    let amplitudes = amplitudes.as_array();

    let out = konnoohmachi_smooth(frequencies, amplitudes, bandwidth);
    out.into_pyarray(py)
}

/// Index of the first zero frequency, or [`NO_ZERO_FREQUENCY`] if there is none.
fn find_zero_frequency(frequencies: &ArrayView1<f64>) -> usize {
    frequencies
        .iter()
        .position(|&v| v == 0.0)
        .unwrap_or(NO_ZERO_FREQUENCY)
}

/// Writes the unnormalized Konno-Ohmachi window for `corner_frequency` into `window` and returns
/// the normalization factor (the window sum).
///
/// `index_frequency` is the position of `corner_frequency` itself, where `sin(x)/x` is the
/// removable singularity `x -> 0` and the window is defined to be 1. A zero frequency contributes
/// nothing and is forced to 0 -- this happens after the singularity fix-up, so a zero corner
/// frequency yields a zero weight for itself.
fn fill_window(
    window: &mut ArrayViewMut1<f64>,
    frequencies: &ArrayView1<f64>,
    corner_frequency: f64,
    bandwidth: f64,
    index_frequency: usize,
    zero_frequency_index: usize,
) -> f64 {
    Zip::from(&mut *window)
        .and(frequencies)
        .for_each(|weight, &frequency| {
            let x = f64::log10(frequency / corner_frequency) * bandwidth;
            *weight = f64::powi(f64::sin(x) / x, 4);
        });

    window[index_frequency] = 1.;

    if zero_frequency_index != NO_ZERO_FREQUENCY {
        window[zero_frequency_index] = 0.;
    }

    window.sum()
}

/// Smooth a spectrum provided as two one-dimensional vectors containing
/// `frequencies` and `amplitudes`
///
/// Every window is built on the fly and discarded, which keeps memory at O(n) but repeats the
/// expensive window construction on every call. To smooth many spectra that share one frequency
/// axis, use [`Smoother`] instead, which builds the windows once.
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

    let frequencies = frequencies
        .into_dimensionality::<Ix1>()
        .expect("frequencies have to be one-dimensional.");
    let amplitudes = amplitudes
        .into_dimensionality::<Ix1>()
        .expect("amplitudes have to be one-dimensional.");

    let zero_frequency_index = find_zero_frequency(&frequencies);

    let mut out = Array1::<f64>::zeros(frequencies.len());
    let mut window = Array1::<f64>::zeros(frequencies.len());

    for (index_frequency, &corner_frequency) in frequencies.iter().enumerate() {
        let normalization = fill_window(
            &mut window.view_mut(),
            &frequencies,
            corner_frequency,
            bandwidth,
            index_frequency,
            zero_frequency_index,
        );

        out[index_frequency] = window.dot(&amplitudes) / normalization;
    }

    out
}

/// A set of pre-computed Konno-Ohmachi smoothing windows for one frequency axis and bandwidth.
///
/// Building the windows is the expensive part of the algorithm; applying them is a single
/// matrix-vector product. Constructing a `Smoother` once and reusing it therefore pays off as soon
/// as more than one spectrum is smoothed on the same frequency axis.
///
/// The windows are stored densely as an `n x n` matrix, so memory grows quadratically: roughly
/// `8 * n^2` bytes, i.e. 8 MB for 1000 samples but 8 GB for 32768. Check [`Smoother::nbytes`]
/// before caching a long frequency axis, and fall back to [`konnoohmachi_smooth`] when the matrix
/// would not fit.
#[pyclass]
pub struct Smoother {
    /// Row `i` is the normalized window centered on `frequencies[i]`.
    windows: Array2<f64>,
}

impl Smoother {
    /// Pre-computes the smoothing windows for `frequencies` and `bandwidth`.
    pub fn new(frequencies: ArrayView1<f64>, bandwidth: f64) -> Self {
        assert!(bandwidth > 0.0, "bandwidth has to be greater than 0.");

        let n_frequencies = frequencies.len();
        let zero_frequency_index = find_zero_frequency(&frequencies);
        let mut windows = Array2::<f64>::zeros((n_frequencies, n_frequencies));

        for (index_frequency, &corner_frequency) in frequencies.iter().enumerate() {
            let mut window = windows.row_mut(index_frequency);
            let normalization = fill_window(
                &mut window,
                &frequencies,
                corner_frequency,
                bandwidth,
                index_frequency,
                zero_frequency_index,
            );
            window.map_inplace(|weight| *weight /= normalization);
        }

        Self { windows }
    }

    /// Smooths `amplitudes` with the cached windows.
    pub fn smooth(&self, amplitudes: ArrayView1<f64>) -> Array1<f64> {
        assert_eq!(
            self.len(),
            amplitudes.len(),
            "amplitudes and frequencies have to have equal length."
        );

        self.windows.dot(&amplitudes)
    }

    /// Number of frequencies this smoother was built for.
    pub fn len(&self) -> usize {
        self.windows.nrows()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Memory held by the cached windows, in bytes.
    pub fn nbytes(&self) -> usize {
        self.windows.len() * std::mem::size_of::<f64>()
    }
}

/// Python interface: pre-computed smoothing windows for one frequency axis.
#[pymethods]
impl Smoother {
    #[new]
    fn py_new(frequencies: PyReadonlyArrayDyn<'_, f64>, bandwidth: f64) -> PyResult<Self> {
        if bandwidth <= 0.0 {
            return Err(PyValueError::new_err("bandwidth has to be greater than 0."));
        }

        let frequencies = frequencies.as_array();
        let frequencies = frequencies
            .into_dimensionality::<Ix1>()
            .map_err(|_| PyValueError::new_err("frequencies have to be one-dimensional."))?;

        Ok(Self::new(frequencies, bandwidth))
    }

    /// Smooth `amplitudes` using the cached windows.
    #[pyo3(name = "smooth")]
    fn py_smooth<'py>(
        &self,
        py: Python<'py>,
        amplitudes: PyReadonlyArrayDyn<'py, f64>,
    ) -> PyResult<Bound<'py, PyArray1<f64>>> {
        let amplitudes = amplitudes.as_array();
        let amplitudes = amplitudes
            .into_dimensionality::<Ix1>()
            .map_err(|_| PyValueError::new_err("amplitudes have to be one-dimensional."))?;

        if amplitudes.len() != self.len() {
            return Err(PyValueError::new_err(
                "amplitudes and frequencies have to have equal length.",
            ));
        }

        Ok(self.smooth(amplitudes).into_pyarray(py))
    }

    /// Memory held by the cached windows, in bytes.
    #[getter(nbytes)]
    fn py_nbytes(&self) -> usize {
        self.nbytes()
    }

    fn __len__(&self) -> usize {
        self.len()
    }
}

#[cfg(test)]
mod tests {
    use ndarray::{Array1, ArrayView1};

    use crate::{konnoohmachi_smooth, Smoother};

    /// Relative tolerance between the cached and the uncached path. Both apply the same operations
    /// but accumulate the normalization in a different order, so results agree only to rounding.
    const TOLERANCE: f64 = 1e-12;

    /// Compares two spectra, treating NaN as matching NaN.
    ///
    /// A zero corner frequency makes `log10(f / 0) = inf` and hence `sin(inf) / inf = NaN`, which
    /// poisons that row's normalization and yields NaN for the zero-frequency bin. That predates
    /// the window cache; these tests pin the cached path to the uncached one, NaN included, rather
    /// than quietly diverge from it.
    fn assert_all_close(left: &ArrayView1<f64>, right: &ArrayView1<f64>) {
        assert_eq!(left.len(), right.len());
        for (index, (&l, &r)) in left.iter().zip(right.iter()).enumerate() {
            if l.is_nan() && r.is_nan() {
                continue;
            }
            let deviation = (l - r).abs() / r.abs().max(1.0);
            assert!(
                deviation < TOLERANCE,
                "element {index} differs: {l} vs {r} (relative deviation {deviation})"
            );
        }
    }

    fn amplitudes(n_samples: usize) -> Array1<f64> {
        // Deterministic but not smooth, so that a broken window shows up in the result.
        Array1::from_iter((0..n_samples).map(|i| ((i * 37) % 13) as f64 + 1.0))
    }

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

    #[test]
    fn test_cached_matches_uncached_with_zero_frequency() {
        let frequencies = Array1::<f64>::linspace(0.0, 10.0, 128);
        let amplitudes = amplitudes(128);
        let bandwidth = 40.0;

        let uncached = konnoohmachi_smooth(
            frequencies.view().into_dyn(),
            amplitudes.view().into_dyn(),
            bandwidth,
        );
        let cached = Smoother::new(frequencies.view(), bandwidth).smooth(amplitudes.view());

        assert_all_close(&cached.view(), &uncached.view());
    }

    #[test]
    fn test_cached_matches_uncached_without_zero_frequency() {
        let frequencies = Array1::<f64>::linspace(1.0, 10.0, 128);
        let amplitudes = amplitudes(128);
        let bandwidth = 40.0;

        let uncached = konnoohmachi_smooth(
            frequencies.view().into_dyn(),
            amplitudes.view().into_dyn(),
            bandwidth,
        );
        let cached = Smoother::new(frequencies.view(), bandwidth).smooth(amplitudes.view());

        assert_all_close(&cached.view(), &uncached.view());
    }

    #[test]
    fn test_smoother_is_reusable() {
        let frequencies = Array1::<f64>::linspace(0.0, 10.0, 64);
        let bandwidth = 40.0;
        let smoother = Smoother::new(frequencies.view(), bandwidth);

        // The same smoother has to keep producing the uncached result for different amplitudes.
        for offset in 0..3 {
            let amplitudes = amplitudes(64) + offset as f64;
            let uncached = konnoohmachi_smooth(
                frequencies.view().into_dyn(),
                amplitudes.view().into_dyn(),
                bandwidth,
            );
            let cached = smoother.smooth(amplitudes.view());
            assert_all_close(&cached.view(), &uncached.view());
        }
    }

    #[test]
    fn test_windows_are_normalized() {
        let frequencies = Array1::<f64>::linspace(1.0, 10.0, 32);
        let smoother = Smoother::new(frequencies.view(), 40.0);

        // A flat spectrum has to come out flat: every window sums to 1.
        let smoothed = smoother.smooth(Array1::<f64>::ones(32).view());
        assert_all_close(&smoothed.view(), &Array1::<f64>::ones(32).view());
    }

    #[test]
    fn test_nbytes_reports_dense_matrix() {
        let frequencies = Array1::<f64>::linspace(1.0, 10.0, 100);
        let smoother = Smoother::new(frequencies.view(), 40.0);
        assert_eq!(smoother.len(), 100);
        assert_eq!(smoother.nbytes(), 100 * 100 * 8);
    }

    #[test]
    #[should_panic(expected = "bandwidth has to be greater than 0.")]
    fn test_zero_bandwidth_panics() {
        let frequencies = Array1::<f64>::linspace(1.0, 10.0, 8);
        Smoother::new(frequencies.view(), 0.0);
    }

    #[test]
    #[should_panic(expected = "equal length")]
    fn test_length_mismatch_panics() {
        let frequencies = Array1::<f64>::linspace(1.0, 10.0, 8);
        let smoother = Smoother::new(frequencies.view(), 40.0);
        smoother.smooth(Array1::<f64>::ones(9).view());
    }
}
