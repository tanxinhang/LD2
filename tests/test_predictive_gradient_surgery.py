import torch

from uav_isac.prediction.gradient_surgery import physical_anchor_pcgrad


def test_physical_anchor_pcgrad_removes_opposing_structure_component():
    parameter = torch.nn.Parameter(torch.tensor([1.0, 1.0]))
    structure = parameter[0] - parameter[1]
    physical = -parameter[0]
    cosine = physical_anchor_pcgrad(structure, physical, [parameter])
    assert cosine < 0.0
    physical_gradient = torch.tensor([-1.0, 0.0])
    assert torch.dot(parameter.grad, physical_gradient) >= 0.0
    torch.testing.assert_close(parameter.grad, torch.tensor([-1.0, -1.0]))


def test_physical_anchor_pcgrad_preserves_compatible_sum():
    parameter = torch.nn.Parameter(torch.tensor([2.0, 3.0]))
    structure = parameter.sum()
    physical = 2.0 * parameter.sum()
    cosine = physical_anchor_pcgrad(structure, physical, [parameter])
    assert cosine > 0.0
    torch.testing.assert_close(parameter.grad, torch.tensor([3.0, 3.0]))
