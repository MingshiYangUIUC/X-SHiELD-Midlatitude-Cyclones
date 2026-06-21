"""
Author:
    Mingshi Yang (mingshi3@illinois.edu)

Date:
    2026-06-21

Project:
    Characteristics of Midlatitude Cyclones under Climate Change in a Global Storm-Resolving Model

Description:
    Created during Revision 1 to calculate cyclone-centered moisture-transport
    budget diagnostics. The script computes horizontal CIVT decomposition from
    925 to 200 hPa, approximates vertical moisture-flux convergence from 925 and
    200 hPa boundary q*omega terms, and saves composite-mean and sample-level
    diagnostics for each experiment.
"""

import os
import numpy as np
import xarray as xr
from tqdm import tqdm

# input and output directory
composite_dir = '/data/keeling/a/mingshi3/c/xshield/composites'
composite_derived_dir = '/data/keeling/a/mingshi3/c/xshield/Midlatitude-Cyclone-in-X-SHiELD/data/composites'
track_dir = '/data/keeling/a/mingshi3/c/xshield/Midlatitude-Cyclone-in-X-SHiELD/data/tracks'

experiments = [
    'PIRE',  # CTRL
    'PIRE_CO2_1270ppmv',
    'PIRE_PLUS_4K',
    'PIRE_PLUS_4K_CO2_1270ppmv'
]

tk = []
idx = []
for exp in experiments:
    tk.append(np.loadtxt(f'{track_dir}/TK2Y-{exp}/All_Tracks.txt'))
    idx.append(np.loadtxt(f'{track_dir}/TK2Y-{exp}/All_Indexes.txt'))
sels = []
for i in range(4):
    d = tk[i]
    sels.append(np.array((d[:,0]>2020000000)&(d[:,0]<2022000000)&(d[:,1]<65)&(d[:,1]>=30)))


#### changed: use 925 hPa as lower boundary for horizontal IVT/CIVT ####
levels = ['925', '850', '700', '500', '200']
boundary_levels = ['925', '200']
version_tag = '925-200vterm_mc'
#### changed ####

dx, dy = 25000.0, 25000.0  # m
g = 9.80665

region = slice(90-2, 101+2)
R_max = 250

#region = slice(80-2, 121+2)
#R_max = 500
#region = slice(60-2, 141+2)
#R_max = 1000

chunk_size = 2048

#### changed: set to 1/3600 if omega files are Pa hour^-1 ####
omega_to_pa_s = 1.0
#### changed ####

def compute_qv_statistics(u_file, v_file, q_file, sel, region, chunk_size=512):
    """
    Compute experiment-level mean fields needed for the qV decomposition:
      mean(q), mean(u), mean(v), mean(q*u), mean(q*v),
      cov(q,u), cov(q,v)
    """

    u = np.load(u_file, mmap_mode='r')
    v = np.load(v_file, mmap_mode='r')
    q = np.load(q_file, mmap_mode='r')

    sel_idx = np.where(sel)[0]
    nsample = len(sel_idx)
    ny, nx = u[sel_idx[0], region, region].shape

    sum_u  = np.zeros((ny, nx), dtype=np.float64)
    sum_v  = np.zeros((ny, nx), dtype=np.float64)
    sum_q  = np.zeros((ny, nx), dtype=np.float64)
    sum_qu = np.zeros((ny, nx), dtype=np.float64)
    sum_qv = np.zeros((ny, nx), dtype=np.float64)

    for i0 in tqdm(range(0, nsample, chunk_size), leave=False):
        i1 = min(i0 + chunk_size, nsample)

        idx_chunk = sel_idx[i0:i1]

        uc = np.asarray(u[idx_chunk, region, region], dtype=np.float64)
        vc = np.asarray(v[idx_chunk, region, region], dtype=np.float64)
        qc = np.asarray(q[idx_chunk, region, region], dtype=np.float64)

        sum_u  += uc.sum(axis=0)
        sum_v  += vc.sum(axis=0)
        sum_q  += qc.sum(axis=0)
        sum_qu += (qc * uc).sum(axis=0)
        sum_qv += (qc * vc).sum(axis=0)

    u_mean  = sum_u  / nsample
    v_mean  = sum_v  / nsample
    q_mean  = sum_q  / nsample
    qu_mean = sum_qu / nsample
    qv_mean = sum_qv / nsample

    qu_cov = qu_mean - q_mean * u_mean
    qv_cov = qv_mean - q_mean * v_mean

    return {
        'u_mean':  u_mean.astype(np.float32),
        'v_mean':  v_mean.astype(np.float32),
        'q_mean':  q_mean.astype(np.float32),
        'qu_mean': qu_mean.astype(np.float32),
        'qv_mean': qv_mean.astype(np.float32),
        'qu_cov':  qu_cov.astype(np.float32),
        'qv_cov':  qv_cov.astype(np.float32),
        'nsample': nsample,
        'ny': ny,
        'nx': nx,
    }


for i, exp in enumerate(experiments):
    for level in levels:
        print(exp, level)

        u_file = f'{composite_dir}/TK2Y-{exp}_u{level}.npy'
        v_file = f'{composite_dir}/TK2Y-{exp}_v{level}.npy'
        q_file = f'{composite_dir}/TK2Y-{exp}_q{level}.npy'

        file_out = f'{composite_derived_dir}/TK2Y-{exp}_qvterms{R_max}-{level}_{version_tag}.nc'

        if os.path.isfile(file_out):
            print('Destination file exists.')
            continue

        stats = compute_qv_statistics(
            u_file=u_file,
            v_file=v_file,
            q_file=q_file,
            sel=sels[i],
            region=region,
            chunk_size=chunk_size,
        )

        coord_idx = np.arange(region.start, region.stop)
        x_km = (coord_idx - 100) * 25.0
        y_km = -(coord_idx - 100) * 25.0

        ds_out = xr.Dataset(
            data_vars={
                'q_mean':  (('y', 'x'), stats['q_mean']),
                'u_mean':  (('y', 'x'), stats['u_mean']),
                'v_mean':  (('y', 'x'), stats['v_mean']),
                'qu_mean': (('y', 'x'), stats['qu_mean']),
                'qv_mean': (('y', 'x'), stats['qv_mean']),
                'qu_cov':  (('y', 'x'), stats['qu_cov']),
                'qv_cov':  (('y', 'x'), stats['qv_cov']),
            },
            coords={
                'x': x_km,
                'y': y_km,
            },
            attrs={
                'experiment': exp,
                'pressure_level_hPa': level,
                'nsample': stats['nsample'],
                'dx_m': dx,
                'dy_m': dy,
                'description': (
                    'Cyclone-centered mean fields for horizontal moisture-flux decomposition. '
                    'Version uses 925 hPa as lower boundary for integrated CIVT.'
                ),
            }
        )

        encoding = {var: {'dtype': 'float32'} for var in ds_out.data_vars}
        ds_out.to_netcdf(file_out, encoding=encoding)
        print('Saved:', file_out)

#### added: only boundary qomega statistics, no full vertical qomega storage ####
def compute_qomega_boundary_statistics(q_file, omg_file, sel, region, chunk_size=512):
    """
    Compute boundary-level mean(q), mean(omega), mean(q*omega), and cov(q, omega).
    Used only at 925 and 200 hPa for vertical boundary moisture-flux term.
    """

    q = np.load(q_file, mmap_mode='r')
    omg = np.load(omg_file, mmap_mode='r')

    sel_idx = np.where(sel)[0]
    nsample = len(sel_idx)
    ny, nx = q[sel_idx[0], region, region].shape

    sum_q = np.zeros((ny, nx), dtype=np.float64)
    sum_omg = np.zeros((ny, nx), dtype=np.float64)
    sum_qomg = np.zeros((ny, nx), dtype=np.float64)

    for i0 in tqdm(range(0, nsample, chunk_size), leave=False):
        i1 = min(i0 + chunk_size, nsample)

        idx_chunk = sel_idx[i0:i1]

        qc = np.asarray(q[idx_chunk, region, region], dtype=np.float64)
        omgc = np.asarray(omg[idx_chunk, region, region], dtype=np.float64) * omega_to_pa_s

        sum_q += qc.sum(axis=0)
        sum_omg += omgc.sum(axis=0)
        sum_qomg += (qc * omgc).sum(axis=0)

    q_mean = sum_q / nsample
    omg_mean = sum_omg / nsample
    qomg_mean = sum_qomg / nsample
    qomg_cov = qomg_mean - q_mean * omg_mean

    return {
        'q_mean': q_mean.astype(np.float32),
        'omg_mean': omg_mean.astype(np.float32),
        'qomg_mean': qomg_mean.astype(np.float32),
        'qomg_cov': qomg_cov.astype(np.float32),
        'nsample': nsample,
    }


for i, exp in enumerate(experiments):
    #### requested omega file pattern, but only for 925 and 200 hPa ####
    omg_files = [f'{composite_dir}/TK2Y-{exp}_omg{level}.npy' for level in boundary_levels]
    q_files = [f'{composite_dir}/TK2Y-{exp}_q{level}.npy' for level in boundary_levels]

    for level, q_file, omg_file in zip(boundary_levels, q_files, omg_files):
        print('qomega boundary', exp, level)

        file_out = f'{composite_derived_dir}/TK2Y-{exp}_qomega-boundary{R_max}-{level}_{version_tag}.nc'

        if os.path.isfile(file_out):
            print('Destination file exists.')
            continue

        stats = compute_qomega_boundary_statistics(
            q_file=q_file,
            omg_file=omg_file,
            sel=sels[i],
            region=region,
            chunk_size=chunk_size,
        )

        coord_idx = np.arange(region.start, region.stop)
        x_km = (coord_idx - 100) * 25.0
        y_km = -(coord_idx - 100) * 25.0

        ds_out = xr.Dataset(
            data_vars={
                'q_mean': (('y', 'x'), stats['q_mean']),
                'omg_mean': (('y', 'x'), stats['omg_mean']),
                'qomg_mean': (('y', 'x'), stats['qomg_mean']),
                'qomg_cov': (('y', 'x'), stats['qomg_cov']),
            },
            coords={
                'x': x_km,
                'y': y_km,
            },
            attrs={
                'experiment': exp,
                'pressure_level_hPa': level,
                'nsample': stats['nsample'],
                'omega_to_pa_s': omega_to_pa_s,
                'description': (
                    'Boundary qomega statistics for approximated vertical moisture-flux convergence. '
                    'Only 925 and 200 hPa are used.'
                ),
            }
        )

        encoding = {var: {'dtype': 'float32'} for var in ds_out.data_vars}
        ds_out.to_netcdf(file_out, encoding=encoding)
        print('Saved:', file_out)
#### added ####

p = np.array([float(lv) for lv in levels]) * 100.0  # Pa, descending 925 -> 200


def open_qvterms(exp, level):
    #### changed: read new 925-lower-boundary qv files ####
    file = f'{composite_derived_dir}/TK2Y-{exp}_qvterms{R_max}-{level}_{version_tag}.nc'
    #### changed ####
    return xr.open_dataset(file)


#### added ####
def open_qomega_boundary(exp, level):
    file = f'{composite_derived_dir}/TK2Y-{exp}_qomega-boundary{R_max}-{level}_{version_tag}.nc'
    return xr.open_dataset(file)
#### added ####


def div2d(fx, fy):
    return np.gradient(fx, dx, axis=-1) - np.gradient(fy, dy, axis=-2)


def vertical_integrate(flux_levels):
    """
    flux_levels shape: level, y, x
    IVT = -1/g int flux dp
    negative sign because p is descending from 925 to 200 hPa.
    """
    return -np.trapz(flux_levels, p, axis=0) / g


#### added: boundary vertical convergence from lower and upper qomega terms ####
def boundary_vconv(qomg_lower, qomg_upper):
    """
    VCONV = -1/g * [(q omega)_925 - (q omega)_200]
    Positive means vertical moisture-flux convergence into the 925-200 hPa layer.
    """
    return -(qomg_lower - qomg_upper) / g
#### added ####


def calc_moisture_budget_925vterm(exp):
    print(exp)

    # horizontal qV terms
    thermo_u_all, thermo_v_all = [], []
    dynamic_u_all, dynamic_v_all = [], []
    nonlinear_u_all, nonlinear_v_all = [], []
    cov_u_all, cov_v_all = [], []
    total_u_all, total_v_all = [], []

    for level in levels:
        ds0 = open_qvterms('PIRE', level)
        ds1 = open_qvterms(exp, level)

        dq = ds1['q_mean'] - ds0['q_mean']
        du = ds1['u_mean'] - ds0['u_mean']
        dv = ds1['v_mean'] - ds0['v_mean']

        thermo_u = ds0['u_mean'] * dq
        dynamic_u = ds0['q_mean'] * du
        nonlinear_u = dq * du
        cov_u = ds1['qu_cov'] - ds0['qu_cov']
        total_u = ds1['qu_mean'] - ds0['qu_mean']

        thermo_v = ds0['v_mean'] * dq
        dynamic_v = ds0['q_mean'] * dv
        nonlinear_v = dq * dv
        cov_v = ds1['qv_cov'] - ds0['qv_cov']
        total_v = ds1['qv_mean'] - ds0['qv_mean']

        thermo_u_all.append(thermo_u.values)
        thermo_v_all.append(thermo_v.values)
        dynamic_u_all.append(dynamic_u.values)
        dynamic_v_all.append(dynamic_v.values)
        nonlinear_u_all.append(nonlinear_u.values)
        nonlinear_v_all.append(nonlinear_v.values)
        cov_u_all.append(cov_u.values)
        cov_v_all.append(cov_v.values)
        total_u_all.append(total_u.values)
        total_v_all.append(total_v.values)

        x = ds0['x'].values
        y = ds0['y'].values

        ds0.close()
        ds1.close()

    # stack to level, y, x
    thermo_u_all = np.stack(thermo_u_all, axis=0)
    thermo_v_all = np.stack(thermo_v_all, axis=0)
    dynamic_u_all = np.stack(dynamic_u_all, axis=0)
    dynamic_v_all = np.stack(dynamic_v_all, axis=0)
    nonlinear_u_all = np.stack(nonlinear_u_all, axis=0)
    nonlinear_v_all = np.stack(nonlinear_v_all, axis=0)
    cov_u_all = np.stack(cov_u_all, axis=0)
    cov_v_all = np.stack(cov_v_all, axis=0)
    total_u_all = np.stack(total_u_all, axis=0)
    total_v_all = np.stack(total_v_all, axis=0)

    # horizontal CIVT
    IVTu_thermo = vertical_integrate(thermo_u_all)
    IVTv_thermo = vertical_integrate(thermo_v_all)
    IVTu_dynamic = vertical_integrate(dynamic_u_all)
    IVTv_dynamic = vertical_integrate(dynamic_v_all)
    IVTu_nonlinear = vertical_integrate(nonlinear_u_all)
    IVTv_nonlinear = vertical_integrate(nonlinear_v_all)
    IVTu_cov = vertical_integrate(cov_u_all)
    IVTv_cov = vertical_integrate(cov_v_all)
    IVTu_total = vertical_integrate(total_u_all)
    IVTv_total = vertical_integrate(total_v_all)

    CIVT_thermo = -div2d(IVTu_thermo, IVTv_thermo)
    CIVT_dynamic = -div2d(IVTu_dynamic, IVTv_dynamic)
    CIVT_nonlinear = -div2d(IVTu_nonlinear, IVTv_nonlinear)
    CIVT_cov = -div2d(IVTu_cov, IVTv_cov)
    CIVT_total = -div2d(IVTu_total, IVTv_total)

    CIVT_sum = CIVT_thermo + CIVT_dynamic + CIVT_nonlinear + CIVT_cov
    CIVT_residual = CIVT_total - CIVT_sum

    #### added: vertical term using only 925 and 200 hPa qomega ####
    ds0_low = open_qomega_boundary('PIRE', boundary_levels[0])
    ds1_low = open_qomega_boundary(exp, boundary_levels[0])
    ds0_top = open_qomega_boundary('PIRE', boundary_levels[1])
    ds1_top = open_qomega_boundary(exp, boundary_levels[1])

    low_term = ds0_low['omg_mean'] * (ds1_low['q_mean'] - ds0_low['q_mean'])
    top_term = ds0_top['omg_mean'] * (ds1_top['q_mean'] - ds0_top['q_mean'])

    Vthermo = -(low_term - top_term) / g

    #print('mean omega0 925:', float(ds0_low['omg_mean'].mean()))
    #print('mean omega1 925:', float(ds1_low['omg_mean'].mean()))
    #print('mean dq 925:', float((ds1_low['q_mean'] - ds0_low['q_mean']).mean()))
    #print('mean low thermo numerator:', float(low_term.mean()))
    #print('mean top thermo numerator:', float(top_term.mean()))
    #print('mean VCONV thermo:', float(Vthermo.mean()))

    def qomega_terms(ds0, ds1):
        dq = ds1['q_mean'] - ds0['q_mean']
        domg = ds1['omg_mean'] - ds0['omg_mean']

        thermo = ds0['omg_mean'] * dq          # q contribution
        dynamic = ds0['q_mean'] * domg         # omega contribution
        nonlinear = dq * domg
        covariance = ds1['qomg_cov'] - ds0['qomg_cov']
        total = ds1['qomg_mean'] - ds0['qomg_mean']

        return thermo, dynamic, nonlinear, covariance, total

    low_thermo, low_dynamic, low_nonlinear, low_cov, low_total = qomega_terms(ds0_low, ds1_low)
    top_thermo, top_dynamic, top_nonlinear, top_cov, top_total = qomega_terms(ds0_top, ds1_top)

    VCONV_thermo_q = boundary_vconv(low_thermo, top_thermo)
    VCONV_dynamic_omega = boundary_vconv(low_dynamic, top_dynamic)
    VCONV_nonlinear = boundary_vconv(low_nonlinear, top_nonlinear)
    VCONV_covariance = boundary_vconv(low_cov, top_cov)
    VCONV_total = boundary_vconv(low_total, top_total)

    VCONV_sum = VCONV_thermo_q + VCONV_dynamic_omega + VCONV_nonlinear + VCONV_covariance
    VCONV_residual = VCONV_total - VCONV_sum

    ds0_low.close()
    ds1_low.close()
    ds0_top.close()
    ds1_top.close()

    # combined horizontal + approximate vertical moisture-flux convergence
    MFC_total = CIVT_total + VCONV_total
    MFC_thermo_q = CIVT_thermo + VCONV_thermo_q

    # horizontal wind dynamic + vertical omega dynamic
    MFC_dynamic_wind_omega = CIVT_dynamic + VCONV_dynamic_omega

    MFC_nonlinear = CIVT_nonlinear + VCONV_nonlinear
    MFC_covariance = CIVT_cov + VCONV_covariance
    MFC_sum = MFC_thermo_q + MFC_dynamic_wind_omega + MFC_nonlinear + MFC_covariance
    MFC_residual = MFC_total - MFC_sum
    #### added ####

    ds_out = xr.Dataset(
        data_vars={
            # original-style horizontal CIVT
            'CIVT_total': (('y', 'x'), CIVT_total.astype(np.float32)),
            'CIVT_thermo': (('y', 'x'), CIVT_thermo.astype(np.float32)),
            'CIVT_dynamic': (('y', 'x'), CIVT_dynamic.astype(np.float32)),
            'CIVT_nonlinear': (('y', 'x'), CIVT_nonlinear.astype(np.float32)),
            'CIVT_covariance': (('y', 'x'), CIVT_cov.astype(np.float32)),
            'CIVT_sum': (('y', 'x'), CIVT_sum.astype(np.float32)),
            'CIVT_residual': (('y', 'x'), CIVT_residual.astype(np.float32)),

            #### added: vertical boundary term ####
            'VCONV_total': (('y', 'x'), VCONV_total.values.astype(np.float32)),
            'VCONV_thermo_q': (('y', 'x'), VCONV_thermo_q.values.astype(np.float32)),
            'VCONV_dynamic_omega': (('y', 'x'), VCONV_dynamic_omega.values.astype(np.float32)),
            'VCONV_nonlinear': (('y', 'x'), VCONV_nonlinear.values.astype(np.float32)),
            'VCONV_covariance': (('y', 'x'), VCONV_covariance.values.astype(np.float32)),
            'VCONV_sum': (('y', 'x'), VCONV_sum.values.astype(np.float32)),
            'VCONV_residual': (('y', 'x'), VCONV_residual.values.astype(np.float32)),

            # combined horizontal + approximate vertical convergence
            'MFC_total': (('y', 'x'), MFC_total.values.astype(np.float32)),
            'MFC_thermo_q': (('y', 'x'), MFC_thermo_q.values.astype(np.float32)),
            'MFC_dynamic_wind_omega': (('y', 'x'), MFC_dynamic_wind_omega.values.astype(np.float32)),
            'MFC_nonlinear': (('y', 'x'), MFC_nonlinear.values.astype(np.float32)),
            'MFC_covariance': (('y', 'x'), MFC_covariance.values.astype(np.float32)),
            'MFC_sum': (('y', 'x'), MFC_sum.values.astype(np.float32)),
            'MFC_residual': (('y', 'x'), MFC_residual.values.astype(np.float32)),
            #### added ####

            # IVT terms
            'IVTu_total': (('y', 'x'), IVTu_total.astype(np.float32)),
            'IVTv_total': (('y', 'x'), IVTv_total.astype(np.float32)),
            'IVTu_thermo': (('y', 'x'), IVTu_thermo.astype(np.float32)),
            'IVTv_thermo': (('y', 'x'), IVTv_thermo.astype(np.float32)),
            'IVTu_dynamic': (('y', 'x'), IVTu_dynamic.astype(np.float32)),
            'IVTv_dynamic': (('y', 'x'), IVTv_dynamic.astype(np.float32)),
        },
        coords={
            'x': x,
            'y': y,
        },
        attrs={
            'experiment': exp,
            'reference_experiment': 'PIRE',
            'description': (
                'Version 925vterm. Horizontal CIVT is integrated from 925 to 200 hPa. '
                'Vertical moisture-flux convergence is approximated only from boundary levels: '
                'VCONV = -1/g * [(q*omega)_925 - (q*omega)_200]. '
                'MFC = CIVT + VCONV. Positive values mean moisture-flux convergence. '
                'Thermo_q isolates delta q. Dynamic isolates horizontal wind changes in CIVT '
                'and omega changes in VCONV.'
            ),
            'CIVT_units': 'kg m-2 s-1',
            'VCONV_units': 'kg m-2 s-1',
            'MFC_units': 'kg m-2 s-1',
            'IVT_units': 'kg m-1 s-1',
            'pressure_levels_hPa_for_horizontal_CIVT': ','.join(levels),
            'boundary_levels_hPa_for_vertical_term': ','.join(boundary_levels),
            'omega_to_pa_s': omega_to_pa_s,
            'dx_m': dx,
            'dy_m': dy,
        }
    )

    #### changed: new output filename ####
    file_out = f'{composite_derived_dir}/TK2Y-{exp}_moisture-transport-budget-{R_max}_{version_tag}.nc'
    #### changed ####

    encoding = {var: {'dtype': 'float32'} for var in ds_out.data_vars}
    ds_out.to_netcdf(file_out, encoding=encoding)

    print('Saved:', file_out)
    #print('Max CIVT residual:', np.nanmax(np.abs(CIVT_residual)))
    #print('Max VCONV residual:', float(np.nanmax(np.abs(VCONV_residual))))
    #print('Max MFC residual:', float(np.nanmax(np.abs(MFC_residual))))

    return ds_out


for exp in experiments[1:]:
    ds_budget = calc_moisture_budget_925vterm(exp)

#quit()

def calc_instantaneous_mfc_for_exp(exp, sel):
    print('MFC', exp)

    #### changed: new filename ####
    file_out = f'{composite_derived_dir}/TK2Y-{exp}_ivt{R_max}_{version_tag}.nc'
    #### changed ####

    if os.path.isfile(file_out):
        print('Destination file exists.')
        return

    
    q_files = [f'{composite_dir}/TK2Y-{exp}_q{level}.npy' for level in levels]
    u_files = [f'{composite_dir}/TK2Y-{exp}_u{level}.npy' for level in levels]
    v_files = [f'{composite_dir}/TK2Y-{exp}_v{level}.npy' for level in levels]

    q_mmaps = [np.load(f, mmap_mode='r') for f in q_files]
    u_mmaps = [np.load(f, mmap_mode='r') for f in u_files]
    v_mmaps = [np.load(f, mmap_mode='r') for f in v_files]

    
    omg_files = [f'{composite_dir}/TK2Y-{exp}_omg{level}.npy' for level in boundary_levels]
    q_boundary_files = [f'{composite_dir}/TK2Y-{exp}_q{level}.npy' for level in boundary_levels]

    q_boundary_mmaps = [np.load(f, mmap_mode='r') for f in q_boundary_files]
    omg_mmaps = [np.load(f, mmap_mode='r') for f in omg_files]

    sel_idx = np.where(sel)[0]
    nsample = len(sel_idx)

    ny = region.stop - region.start
    nx = region.stop - region.start

    civt_out = np.zeros((nsample, ny, nx), dtype=np.float32)
    ivtu_out = np.zeros((nsample, ny, nx), dtype=np.float32)
    ivtv_out = np.zeros((nsample, ny, nx), dtype=np.float32)

    #### added ####
    vconv_out = np.zeros((nsample, ny, nx), dtype=np.float32)
    mfc_out = np.zeros((nsample, ny, nx), dtype=np.float32)
    #### added ####

    p = np.array([float(level) for level in levels]) * 100.0

    for i0 in tqdm(range(0, nsample, chunk_size)):
        i1 = min(i0 + chunk_size, nsample)
        idx_chunk = sel_idx[i0:i1]

        q_all, u_all, v_all = [], [], []

        for k, level in enumerate(levels):
            q = np.asarray(q_mmaps[k][idx_chunk, region, region], dtype=np.float64)
            u = np.asarray(u_mmaps[k][idx_chunk, region, region], dtype=np.float64)
            v = np.asarray(v_mmaps[k][idx_chunk, region, region], dtype=np.float64)

            q_all.append(q)
            u_all.append(u)
            v_all.append(v)

        q_all = np.stack(q_all, axis=0)
        u_all = np.stack(u_all, axis=0)
        v_all = np.stack(v_all, axis=0)

        # horizontal IVT/CIVT from 925 to 200 hPa
        ivtu = -np.trapz(q_all * u_all, p, axis=0) / g
        ivtv = -np.trapz(q_all * v_all, p, axis=0) / g

        div_ivt = (
            np.gradient(ivtu, dx, axis=2) -
            np.gradient(ivtv, dy, axis=1)
        )

        civt = -div_ivt

        #### added: vertical term only from 925 and 200 hPa ####
        q_low = np.asarray(q_boundary_mmaps[0][idx_chunk, region, region], dtype=np.float64)
        q_top = np.asarray(q_boundary_mmaps[1][idx_chunk, region, region], dtype=np.float64)

        omg_low = np.asarray(omg_mmaps[0][idx_chunk, region, region], dtype=np.float64) * omega_to_pa_s
        omg_top = np.asarray(omg_mmaps[1][idx_chunk, region, region], dtype=np.float64) * omega_to_pa_s

        qomg_low = q_low * omg_low
        qomg_top = q_top * omg_top

        vconv = -(qomg_low - qomg_top) / g
        mfc = civt + vconv
        #### added ####

        ivtu_out[i0:i1] = ivtu.astype(np.float32)
        ivtv_out[i0:i1] = ivtv.astype(np.float32)
        civt_out[i0:i1] = civt.astype(np.float32)

        #### added ####
        vconv_out[i0:i1] = vconv.astype(np.float32)
        mfc_out[i0:i1] = mfc.astype(np.float32)
        #### added ####

    x = (np.arange(region.start, region.stop) - 100) * 25.0
    y = -(np.arange(region.start, region.stop) - 100) * 25.0
    xx, yy = np.meshgrid(x, y)
    mask_R_max = (xx**2 + yy**2) <= R_max**2

    print(f'{R_max}-km valid grid points:', mask_R_max.sum())

    ivtu_mean = ivtu_out[:, mask_R_max].mean(axis=1)
    ivtv_mean = ivtv_out[:, mask_R_max].mean(axis=1)
    civt_mean = civt_out[:, mask_R_max].mean(axis=1)

    #### added ####
    vconv_mean = vconv_out[:, mask_R_max].mean(axis=1)
    mfc_mean = mfc_out[:, mask_R_max].mean(axis=1)
    #### added ####

    ds = xr.Dataset(
        data_vars={
            'IVTu': (('sample',), ivtu_mean.astype(np.float32)),
            'IVTv': (('sample',), ivtv_mean.astype(np.float32)),
            'CIVT': (('sample',), civt_mean.astype(np.float32)),

            #### added ####
            'CIVT_horizontal': (('sample',), civt_mean.astype(np.float32)),
            'VCONV_vertical_925_200': (('sample',), vconv_mean.astype(np.float32)),
            'MFC_CIVT_plus_VCONV': (('sample',), mfc_mean.astype(np.float32)),
            #### added ####
        },
        coords={
            'sample': np.arange(nsample),
        },
        attrs={
            'experiment': exp,
            'description': (
                f'Version 925vterm. {R_max}-km radius area-mean instantaneous diagnostics. '
                'Horizontal CIVT is integrated from 925 to 200 hPa. '
                'Vertical moisture-flux convergence is approximated using only 925 and 200 hPa: '
                'VCONV = -1/g * [(q*omega)_925 - (q*omega)_200]. '
                'MFC = CIVT + VCONV. Positive values mean moisture-flux convergence.'
            ),
            'IVT_units': 'kg m-1 s-1',
            'CIVT_units': 'kg m-2 s-1',
            'VCONV_units': 'kg m-2 s-1',
            'MFC_units': 'kg m-2 s-1',
            'dx_m': dx,
            'dy_m': dy,
            'pressure_levels_hPa_for_horizontal_CIVT': ','.join(levels),
            'boundary_levels_hPa_for_vertical_term': ','.join(boundary_levels),
            'omega_to_pa_s': omega_to_pa_s,
            'area_average_radius_km': R_max,
        }
    )

    encoding = {var: {'dtype': 'float32'} for var in ds.data_vars}
    ds.to_netcdf(file_out, encoding=encoding)
    print('Saved:', file_out)


for i, exp in enumerate(experiments):
    calc_instantaneous_mfc_for_exp(exp, sels[i])