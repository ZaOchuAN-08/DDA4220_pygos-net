import pytest
import torch
from torch.nn import functional as F
from pygosnet.models import Baseline, DilatedContextBlock, PygosNet
from pygosnet.models.pygosnet import feature_dampening


torch.set_num_threads(4)


@pytest.mark.parametrize('use_dcb', [False, True])
@pytest.mark.parametrize('shape', [(256, 256), (448, 448), (256, 448)])
def test_shape_and_output_contract(use_dcb, shape):
    torch.manual_seed(42)
    model = PygosNet(use_dcb=use_dcb, pretrained=False).eval()
    with torch.no_grad():
        out = model(torch.randn(1, 3, *shape))
    for value in out.values():
        assert value.shape == (1, 1, shape[0] // 8, shape[1] // 8)
        assert torch.isfinite(value).all()
    assert (out['density'] >= 0).all()
    torch.testing.assert_close(out['segmentation'], out['segmentation_logits'].sigmoid())


def test_gate_boundary():
    p = torch.tensor([0.0, 0.4999, 0.5, 1.0], requires_grad=True)
    torch.testing.assert_close(feature_dampening(p), torch.tensor([.1, .1, 1., 1.]))
    assert not feature_dampening(p).requires_grad


def test_dcb_gradients():
    torch.manual_seed(42)
    block = DilatedContextBlock()
    x = torch.randn(2, 512, 8, 12, requires_grad=True)
    y = block(x)
    assert y.shape == x.shape
    y.square().mean().backward()
    assert torch.isfinite(x.grad).all() and x.grad.abs().sum() > 0
    for parameter in block.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all() and parameter.grad.abs().sum() > 0


@pytest.mark.parametrize('use_dcb', [False, True])
def test_joint_backward_and_optimizer_step(use_dcb):
    torch.manual_seed(42)
    model = PygosNet(use_dcb=use_dcb, pretrained=False)
    # Controlled positive bias keeps this gradient test off the dead-ReLU branch.
    with torch.no_grad():
        model.density_head[-2].bias.fill_(0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
    out = model(torch.randn(1, 3, 64, 64))
    loss = F.mse_loss(out['density'], torch.ones_like(out['density']))
    loss = loss + F.binary_cross_entropy_with_logits(
        out['segmentation_logits'], torch.ones_like(out['segmentation']))
    loss.backward()
    modules = [model.encoder, model.density_head, model.segmentation_head]
    if use_dcb:
        modules.extend(model.context.branches)
        modules.append(model.context.project)
    for module in modules:
        grads = [p.grad for p in module.parameters()]
        assert all(g is not None and torch.isfinite(g).all() for g in grads)
        assert sum(g.abs().sum().item() for g in grads) > 0
    before = model.density_head[-2].weight.detach().clone()
    optimizer.step()
    assert not torch.equal(before, model.density_head[-2].weight)


def test_density_gate_does_not_backprop_to_segmentation():
    model = Baseline(pretrained=False)
    with torch.no_grad():
        model.density_head[-2].bias.fill_(.1)
    model(torch.randn(1, 3, 32, 32))['density'].sum().backward()
    assert all(p.grad is None for p in model.segmentation_head.parameters())
    assert model.encoder.features[0].weight.grad is not None


def test_predict_masks_only_in_prediction_api():
    model = Baseline(pretrained=False).eval()
    with torch.no_grad():
        for p in model.segmentation_head.parameters():
            p.zero_()
        model.segmentation_head[-1].bias.fill_(-1.)
        model.density_head[-2].bias.fill_(1.)
    x = torch.zeros(2, 3, 32, 32)
    out = model.predict(x)
    assert (out['density'] > 0).all()
    assert torch.count_nonzero(out['density_masked']) == 0
    torch.testing.assert_close(out['count'], torch.zeros(2))
    with torch.no_grad():
        model.segmentation_head[-1].bias.zero_()  # sigmoid(0) == 0.5 is included
    out = model.predict(x)
    torch.testing.assert_close(out['density_masked'], out['density'])
    torch.testing.assert_close(out['count'], out['density'].sum((1, 2, 3)))
    model.train()
    with pytest.raises(RuntimeError):
        model.predict(x)


def test_non_dcb_modules_are_identical():
    torch.manual_seed(42)
    baseline = Baseline(pretrained=False)
    full = PygosNet(pretrained=False)
    full.load_state_dict(baseline.state_dict(), strict=False)
    for name in ('encoder', 'density_head', 'segmentation_head'):
        for key, value in getattr(baseline, name).state_dict().items():
            torch.testing.assert_close(value, getattr(full, name).state_dict()[key])
    full.context = torch.nn.Identity()
    x = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        a, b = baseline(x), full(x)
    for key in a:
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)


@pytest.mark.parametrize('shape', [(1, 1, 32, 32), (1, 3, 33, 32), (1, 3, 8, 8)])
def test_invalid_input_rejected(shape):
    with pytest.raises(ValueError):
        Baseline(pretrained=False)(torch.zeros(shape))
