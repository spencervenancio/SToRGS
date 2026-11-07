"""Run a pinned upstream seqlet caller with administrative compatibility only."""
import importlib.util
import sys
import types
from pathlib import Path


def load_upstream(directory, figure_directory):
    """Skip unrelated package imports; patch deprecated import and plot paths."""
    directory, figure_directory = Path(directory), Path(figure_directory)
    package = types.ModuleType('modisco')
    package.__path__ = [str(directory / 'modisco')]
    sys.modules['modisco'] = package
    patches = {}
    for name in ['value_provider', 'util', 'core', 'coordproducers']:
        path = directory / 'modisco' / (name + '.py')
        original = path.read_text()
        code = original.replace('from sklearn.neighbors.kde import KernelDensity',
                                'from sklearn.neighbors import KernelDensity')
        code = code.replace('os.makedirs("figures")', f'os.makedirs({str(figure_directory)!r})')
        code = code.replace('("figures/scoredist_" +',
                            f'({str(figure_directory / "scoredist_")!r} +')
        module_name = 'modisco.' + name
        spec = importlib.util.spec_from_file_location(module_name, path)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        setattr(package, name, module)
        exec(compile(code, str(path), 'exec'), module.__dict__)
        patches[name] = (original, code)
    return package.coordproducers, patches


def seqlets_to_tiles(coords, count, length=200, width=20, flank=10):
    """Map only native central cores; preserve signed native coordinates elsewhere."""
    import numpy as np

    mask = np.zeros((count, length // width), dtype=bool)
    for coord in coords:
        start, end = coord.start + flank, coord.end - flank
        if not (0 <= start < end <= length):
            raise ValueError('Invalid native core interval')
        mask[coord.example_idx, start // width:(end - 1) // width + 1] = True
    return mask
