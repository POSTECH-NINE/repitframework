import pytest
import numpy as np
from torch import Tensor
import json
from .fvmn import FVMNDataset

@pytest.fixture
def dummy_dataset_dir(tmp_path):
    # Create dummy npy files for U and T at three time steps
    npy_paths = []
    for var in ["U", "T"]:
        for t in [10.0, 10.01, 10.02]:
            arr = np.ones((40000, 2)) if var == "U" else np.ones(40000)
            npy_path = tmp_path / f"{var}_{t}.npy"
            np.save(npy_path, arr)
            npy_paths.append(npy_path)
    # Create dummy norm_denorm_metrics.json
    metrics = {
        "input_MEAN": [0.0],
        "input_STD": [1.0],
        "label_MEAN": [0.0],
        "label_STD": [1.0],
        "true_residual_mass": 0.0
    }
    metrics_path = tmp_path / "norm_denorm_metrics.json"
    metrics_path.write_text(json.dumps(metrics))
    return tmp_path

@pytest.mark.parametrize("output_dims", ["BD", "BCD", "BCHW"])
def test_fvmn_dataset_init_and_shapes(dummy_dataset_dir, output_dims):
    ds = FVMNDataset(
        start_time=10.0,
        end_time=10.02,
        time_step=0.01,
        dataset_dir=dummy_dataset_dir,  # FIXED
        first_training=True,
        vars_list=["U", "T"],
        extended_vars_list=["U_x", "U_y", "T"],
        dims=2,
        grid_x=200,
        grid_y=200,
        grid_z=1,
        output_dims=output_dims,
        bc_type="enforced",
        do_feature_selection=True
    )
    assert isinstance(ds.inputs, Tensor)
    assert isinstance(ds.labels, Tensor)
    assert len(ds) == ds.inputs.shape[0]
    # Check shapes are not empty
    assert ds.inputs.numel() > 0
    assert ds.labels.numel() > 0
    if output_dims == "BD":
        assert ds.inputs.shape == (80000, 15)
        assert ds.labels.shape == (80000, 3)
    elif output_dims == "BCD":
        assert ds.inputs.shape == (2, 15, 80000)
        assert ds.labels.shape == (2, 3, 80000)
    elif output_dims == "BCHW":
        assert ds.inputs.shape == (2, 15, 200, 200)
        assert ds.labels.shape == (2, 3, 200, 200)

@pytest.mark.parametrize("output_dims", ["BD", "BCD", "BCHW"])
def test_fvmn_dataset_init_and_shapes_no_feature_selection(dummy_dataset_dir, output_dims):
    ds = FVMNDataset(
        start_time=10.0,
        end_time=10.02,
        time_step=0.01,
        dataset_dir=dummy_dataset_dir,  # FIXED
        first_training=True,
        vars_list=["U", "T"],
        extended_vars_list=["U_x", "U_y", "T"],
        dims=2,
        grid_x=200,
        grid_y=200,
        grid_z=1,
        output_dims=output_dims,
        bc_type="enforced",
        do_feature_selection=False
    )
    assert isinstance(ds.inputs, Tensor)
    assert isinstance(ds.labels, Tensor)
    assert len(ds) == ds.inputs.shape[0]
    # Check shapes are not empty
    assert ds.inputs.numel() > 0
    assert ds.labels.numel() > 0
    if output_dims == "BD":
        assert ds.inputs.shape == (80000, 3)
        assert ds.labels.shape == (80000, 3)
    elif output_dims == "BCD":
        assert ds.inputs.shape == (2, 3, 80000)
        assert ds.labels.shape == (2, 3, 80000)
    elif output_dims == "BCHW":
        assert ds.inputs.shape == (2, 3, 200, 200)
        assert ds.labels.shape == (2, 3, 200, 200)

def test_len_getitem_iter_repr(dummy_dataset_dir):
    ds = FVMNDataset(
        start_time=10.0,
        end_time=10.02,
        time_step=0.01,
        dataset_dir=dummy_dataset_dir,  # FIXED
        first_training=True,
        vars_list=["U", "T"],
        extended_vars_list=["U_x", "U_y", "T"],
        dims=2,
        grid_x=200,
        grid_y=200,
        grid_z=1,
        output_dims="BD",
        bc_type="enforced",
        do_feature_selection=True
    )
    # __len__
    assert len(ds) == ds.inputs.shape[0]
    # __getitem__
    x, y = ds[0]
    assert isinstance(x, Tensor)
    assert isinstance(y, Tensor)
    # __iter__
    for i, (xi, yi) in enumerate(ds):
        assert isinstance(xi, Tensor)
        assert isinstance(yi, Tensor)
        if i > 2:
            break
    # __repr__
    s = repr(ds)
    assert "FVMNDataset" in s

def test_normalize_and_denormalize():
    arr = np.array([[1.0, 2.0], [3.0, 4.0]])
    normed, mean, std = FVMNDataset.normalize(arr)
    restored = FVMNDataset.denormalize(normed, mean, std)
    np.testing.assert_allclose(arr, restored)

def test_parse_numpy_scalar_and_vector(tmp_path):
    # Scalar
    arr = np.arange(40000)
    np.save(tmp_path / "scalar.npy", arr)
    out = FVMNDataset.parse_numpy(tmp_path / "scalar.npy", 200, 200)  # FIXED
    assert out.shape == (200, 200)
    # Vector
    arr = np.ones((40000, 2))
    np.save(tmp_path / "vector.npy", arr)
    out = FVMNDataset.parse_numpy(tmp_path / "vector.npy", 200, 200)  # FIXED
    assert out.shape == (200, 200, 2)

def test_add_feature():
    arr = np.arange(16).reshape(4, 4)
    out = FVMNDataset.add_feature(arr, output_dim="BD")  # FIXED
    assert out.shape[0] == 5  # There should be 5 features

def test_hard_contraint_bc():
    ux = np.zeros((4, 4))
    uy = np.zeros((4, 4))
    t = np.ones((4, 4))
    data_list = [t.copy(), ux.copy(), uy.copy()]
    out = FVMNDataset.hard_contraint_bc(data_list, ["T", "U_x", "U_y"], 100, 200)  # FIXED
    assert isinstance(out, list)
    assert out[0].shape == (6, 6)  # After padding

def test_prepare_label_and_input(dummy_dataset_dir):
    ds = FVMNDataset(
        start_time=10.0,
        end_time=10.02,
        time_step=0.01,
        dataset_dir=dummy_dataset_dir,  # FIXED
        first_training=True,
        vars_list=["U", "T"],
        extended_vars_list=["U_x", "U_y", "T"],
        dims=2,
        grid_x=200,
        grid_y=200,
        grid_z=1,
        output_dims="BD",
        bc_type="enforced",
        do_feature_selection=True
    )
    # _prepare_input and _prepare_label are covered by __init__, but you can call them directly:
    arr = ds._prepare_input(10.0)
    assert isinstance(arr, np.ndarray)
    label = ds._prepare_label(10.0)
    assert isinstance(label, np.ndarray)

def test_missing_file_raises(tmp_path):
    # Directory exists but no files
    with pytest.raises(AssertionError):
        FVMNDataset(
            start_time=10.0,
            end_time=10.02,
            time_step=0.01,
            dataset_dir=tmp_path,  # FIXED
            first_training=True,
            vars_list=["U", "T"],
            extended_vars_list=["U_x", "U_y", "T"],
            dims=2,
            grid_x=200,
            grid_y=200,
            grid_z=1,
            output_dims="BD",
            bc_type="enforced",
            do_feature_selection=True
        )