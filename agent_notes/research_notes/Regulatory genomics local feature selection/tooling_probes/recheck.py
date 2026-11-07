import inspect, torch, torch.nn as nn, captum
from tangermeme.deep_lift_shap import deep_lift_shap
sig = inspect.signature(deep_lift_shap); print("deep_lift_shap", tuple(sig.parameters)); print("  defaults:", {k: v.default for k, v in sig.parameters.items() if v.default is not inspect._empty})
from captum.attr import IntegratedGradients, DeepLiftShap
net = nn.Sequential(nn.Conv1d(4, 32, 19, padding=9), nn.ReLU(), nn.AdaptiveMaxPool1d(1), nn.Flatten(), nn.Linear(32, 1)).eval().to("mps")
x = torch.nn.functional.one_hot(torch.randint(0, 4, (8, 200)), 4).permute(0, 2, 1).float().to("mps")
a = IntegratedGradients(net).attribute(x, baselines=torch.zeros_like(x), target=0, n_steps=50); print("captum", captum.__version__, "IG on MPS OK", tuple(a.shape))
a = DeepLiftShap(net).attribute(x, baselines=x[torch.randperm(8)][:4], target=0); print("captum DeepLiftShap on MPS OK", tuple(a.shape))
