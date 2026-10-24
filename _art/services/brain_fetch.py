# ADHD-200 resting-state fMRI (nilearn's copy), the MSDL atlas, and the mean correlation between its regions
import sys
import numpy as np
from nilearn import datasets
from nilearn.connectome import ConnectivityMeasure
from nilearn.maskers import NiftiMapsMasker

data_dir = sys.argv[1]
adhd = datasets.fetch_adhd(n_subjects=10, data_dir=data_dir)
atlas = datasets.fetch_atlas_msdl(data_dir=data_dir)
masker = NiftiMapsMasker(maps_img=atlas.maps, standardize="zscore_sample", detrend=True, low_pass=0.1, high_pass=0.01, t_r=2.5, memory=f"{data_dir}/cache")
pheno = adhd.phenotypic
series, labels = [], []
for func, conf, row in zip(adhd.func, adhd.confounds, range(len(adhd.func))):
    series.append(masker.fit_transform(func, confounds=conf))
    labels.append(int(pheno["adhd"].iloc[row]) if "adhd" in pheno else -1)
print("subjects", len(series), "adhd flags", labels, "series", series[0].shape)
conn = ConnectivityMeasure(kind="correlation", standardize="zscore_sample")
mats = conn.fit_transform([s for s, l in zip(series, labels) if l == 1] or series)
np.savez(f"{data_dir}/connectome.npz", mean=mats.mean(0), coords=np.array(atlas.region_coords), labels=np.array(atlas.labels),
         networks=np.array(atlas.networks), series=series[0])
print("saved", mats.shape)
