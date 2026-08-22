"""Saving and restoring a simulation to HDF5.

Follows the shape of ``impact/archive.py`` in lume-impact: a self-describing HDF5 file
holding the inputs, the outputs and the bunches, from which a working simulator can be
rebuilt.

The lattice is stored as one dict per element, using ImpactX's own ``to_dict()`` /
``KnownElementsList.from_dicts()`` -- but with a workaround that is not optional.
``to_dict()`` returns **radians** for the angle of every type in
:data:`~lume_impactx.elements.DEGREE_ELEMENTS` while ``from_dicts()`` expects
**degrees**, so a naive round-trip silently divides those angles by 57.3: an
``ExactSbend`` built with ``phi=30`` comes back as ``phi=0.0091 rad`` instead of
``0.5236 rad``. Passing ``in_degrees=True`` fixes it -- and that keyword exists *only*
on the affected types, so it has to be applied conditionally.
"""

from __future__ import annotations

import copy
import json
from typing import Any

import numpy as np

from lume_impactx.elements import DEGREE_ELEMENTS, element_type
from lume_impactx.utils import ImpactXRefPart

try:
    from beamphysics import ParticleGroup
except ImportError:  # pragma: no cover
    from pmd_beamphysics import ParticleGroup

__all__ = ["archive", "load_archive", "element_to_dict", "dicts_to_lattice"]

#: Written to the file root so readers can recognise it.
DATA_TYPE = "lume-impactx"
ARCHIVE_VERSION = 1


def element_to_dict(element: Any) -> dict[str, Any]:
    """Serialize one element, working around the degrees/radians bug.

    Parameters
    ----------
    element : impactx.elements.*
        The element to serialize.

    Returns
    -------
    dict
        A dict that ``KnownElementsList.from_dicts`` reconstructs faithfully.
    """
    if element_type(element) in DEGREE_ELEMENTS:
        return dict(element.to_dict(in_degrees=True))
    return dict(element.to_dict())


def dicts_to_lattice(dicts: list[dict[str, Any]]) -> list:
    """Rebuild a lattice from :func:`element_to_dict` output.

    Returns a plain Python list, which is what
    :class:`~lume_impactx.simulator.ImpactXSimulator` keeps as its canonical lattice.
    The elements hold a reference to the temporary ``KnownElementsList`` that made
    them, so they stay valid after it goes out of scope.
    """
    from impactx import elements

    known = elements.KnownElementsList()
    known.from_dicts(dicts)
    return list(known)


def _json_default(value: Any) -> Any:
    """Encode numpy scalars and array-likes; refuse anything else loudly."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if hasattr(value, "to_numpy"):
        return np.asarray(value.to_numpy()).tolist()
    raise TypeError(
        f"Cannot archive a value of type {type(value).__name__}. Matrix-valued "
        "element attributes (LinearMap.R, SpinMap.A) are not supported yet."
    )


def _open(dest: Any, mode: str):
    """Return ``(group, file_to_close)`` for a path or an already-open group."""
    import h5py

    if isinstance(dest, (h5py.File, h5py.Group)):
        return dest, None
    handle = h5py.File(dest, mode)
    return handle, handle


def archive(simulator: Any, dest: Any) -> None:
    """Write a simulator's inputs, outputs and bunches to HDF5.

    Parameters
    ----------
    simulator : ImpactXSimulator
        The simulator to archive. It must have tracked at least once.
    dest : str or pathlib.Path or h5py.Group
        Destination file or an open group to write into.
    """
    import impactx

    import lume_impactx

    group, handle = _open(dest, "w")
    try:
        group.attrs["dataType"] = DATA_TYPE
        group.attrs["archiveVersion"] = ARCHIVE_VERSION
        group.attrs["lume_impactx_version"] = lume_impactx.__version__
        group.attrs["impactx_version"] = impactx.__version__

        inputs = group.create_group("input")
        inputs.attrs["lattice"] = json.dumps(
            [element_to_dict(e) for e in simulator.lattice], default=_json_default
        )
        inputs.attrs["ref"] = json.dumps(simulator.ref, default=_json_default)
        inputs.attrs["settings"] = json.dumps(simulator.settings, default=_json_default)
        inputs.attrs["capture_at"] = json.dumps(list(simulator.capture_at))
        if simulator.initial_particles is not None:
            simulator.initial_particles.write(inputs, name="initial_particles")
        if simulator.ref_origin is not None:
            inputs.attrs["ref_origin"] = json.dumps(
                simulator.ref_origin.__dict__, default=_json_default
            )

        results = simulator.results
        outputs = group.create_group("output")
        outputs.attrs["moments"] = json.dumps(
            {k: float(v) for k, v in results["moments"].items()}
        )
        outputs.attrs["ref_final"] = json.dumps(results["ref_final"].__dict__)
        for key in ("n_particles", "n_steps", "run_time"):
            outputs.attrs[key] = results[key]

        history = results["moments_history"]
        history_group = outputs.create_group("moments_history")
        for column in history.columns:
            history_group.create_dataset(
                column, data=np.asarray(history[column], dtype=float)
            )

        if results["final_particles"] is not None:
            results["final_particles"].write(outputs, name="final_particles")

        # The captured bunches, not just the promise of them: restoring capture_at
        # without these made the loaded simulator claim a probe it could not produce.
        captured = results.get("captured_particles") or {}
        if captured:
            capture_group = outputs.create_group("captured_particles")
            names = []
            for index, (name, bunch) in enumerate(captured.items()):
                if not isinstance(bunch, ParticleGroup):
                    continue  # an unrepresentable bunch, or an ungathered MPI rank
                # By index, because an element name is not a safe HDF5 group name.
                bunch.write(capture_group, name=str(index))
                names.append(name)
            capture_group.attrs["names"] = json.dumps(names)

        # Linear optics, when the lattice produced them. Stored as datasets so a
        # restored simulator generates the same optics:* variables as the original.
        optics = outputs.create_group("optics")
        for key in ("transfer_map", "cumulative_maps", "map_s"):
            if key in results:
                optics.create_dataset(key, data=np.asarray(results[key], dtype=float))
        if "map_names" in results:
            optics.attrs["map_names"] = json.dumps(list(results["map_names"]))
    finally:
        if handle is not None:
            handle.close()


def load_archive(source: Any, track: bool = False) -> Any:
    """Rebuild a simulator from an archive.

    Parameters
    ----------
    source : str or pathlib.Path or h5py.Group
        The archive to read.
    track : bool
        Re-run the simulation on load. When False (the default) the archived results
        are restored as-is, so reading an archive needs no ImpactX run at all.

    Returns
    -------
    ImpactXSimulator
        With its lattice, reference, settings and cached results restored.

    Raises
    ------
    ValueError
        If the file is not a lume-impactx archive.
    """
    import pandas as pd

    from lume_impactx.simulator import ImpactXSimulator

    group, handle = _open(source, "r")
    try:
        if group.attrs.get("dataType") != DATA_TYPE:
            raise ValueError(
                f"Not a {DATA_TYPE} archive: dataType={group.attrs.get('dataType')!r}."
            )

        inputs = group["input"]
        lattice = dicts_to_lattice(json.loads(inputs.attrs["lattice"]))
        ref = json.loads(inputs.attrs["ref"])
        settings = json.loads(inputs.attrs["settings"])
        # Archives written before capture_at existed simply have no captures.
        capture_at = json.loads(inputs.attrs.get("capture_at", "[]"))

        initial_particles = None
        if "initial_particles" in inputs:
            initial_particles = ParticleGroup(h5=inputs["initial_particles"])

        ref_origin = None
        if "ref_origin" in inputs.attrs:
            ref_origin = ImpactXRefPart(**json.loads(inputs.attrs["ref_origin"]))

        outputs = group["output"]
        history = pd.DataFrame(
            {
                name: np.asarray(outputs["moments_history"][name])
                for name in outputs["moments_history"]
            }
        )
        final_particles = None
        if "final_particles" in outputs:
            final_particles = ParticleGroup(h5=outputs["final_particles"])

        captured_particles: dict[str, Any] = {}
        if "captured_particles" in outputs:
            capture_group = outputs["captured_particles"]
            names = json.loads(capture_group.attrs.get("names", "[]"))
            for index, name in enumerate(names):
                key = str(index)
                if key in capture_group:
                    captured_particles[name] = ParticleGroup(h5=capture_group[key])

        optics: dict[str, Any] = {}
        if "optics" in outputs:
            optics_group = outputs["optics"]
            for key in optics_group:
                optics[key] = np.asarray(optics_group[key])
            if "map_names" in optics_group.attrs:
                optics["map_names"] = json.loads(optics_group.attrs["map_names"])

        results = {
            "moments": json.loads(outputs.attrs["moments"]),
            "moments_history": history,
            "final_particles": final_particles,
            "ref_final": ImpactXRefPart(**json.loads(outputs.attrs["ref_final"])),
            "n_particles": int(outputs.attrs["n_particles"]),
            "n_steps": int(outputs.attrs["n_steps"]),
            "run_time": float(outputs.attrs["run_time"]),
            "captured_particles": captured_particles,
            **optics,
        }
    finally:
        if handle is not None:
            handle.close()

    simulator = ImpactXSimulator.__new__(ImpactXSimulator)
    simulator.lattice = lattice
    simulator.ref = ref
    simulator.settings = settings
    simulator.distribution = None
    simulator.npart = None
    simulator.bunch_charge_C = None
    simulator.initial_particles = initial_particles
    simulator.capture_at = capture_at
    simulator.ref_origin = ref_origin
    simulator._results = results
    simulator.track_count = 0

    from lume_impactx.elements import snapshot_lattice

    simulator._initial_lattice = snapshot_lattice(lattice)
    simulator._initial_ref = dict(ref)
    # deepcopy, matching __init__: a shallow dict shares its list values, so mutating
    # settings["n_cell"] -- itself an exposed variable -- would also mutate the baseline
    # that reset() restores from.
    simulator._initial_settings = copy.deepcopy(settings)
    simulator._initial_particles_0 = initial_particles
    simulator._ref_origin_0 = ref_origin

    if track:
        if initial_particles is None:
            raise ValueError(
                "This archive was seeded from a distribution, which is not stored "
                "(it is not serializable). Re-tracking would need the distribution "
                "back; load with track=False, or rebuild the simulator by hand."
            )
        simulator.track()
    return simulator
