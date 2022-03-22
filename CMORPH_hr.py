
import matplotlib.pyplot as plt
import iris
import iris.analysis.cartography
import numpy as np
import cf_units

def subdomain_constraint(cube, subdomain):
    """Get subdomain of cube

    Arguments:
        cube (iris.Cube): Input cube to extract subdomain from
        subdomain (list): List of coordinates [min lon, max lon, min lat, max lat]

    Returns:
        Iris cube

    """
    if isinstance(subdomain, list):
        # Extract subdomain for rectangular domains
        subcube = cube.intersection(longitude=subdomain[:2],
                                    latitude=subdomain[2:])
        return subcube
    elif isinstance(subdomain, str):
        # Extract subdomain for irregular areas based on shapefiles
        return cube
    else:
        # Raise error if other types of subdomain are passed.
        raise ValueError("Subdomain is type {}.  Should be list or str.".format(type(subdomain)))

DOM_LAT = [-15.5, -15]
DOM_LON = [28, 28.5]
DOM_NAME = 'Lusaka'

CMORPH_hr=f"/scratch/sburgan/obs/CMORPH_V1.0_ADJ_8km-1hr_2009.nc"

cs = iris.coord_systems.GeogCS(iris.fileformats.pp.EARTH_RADIUS)

# load data and then concatenate
print('\33[93m Loading pr observations\33[00m')
data = {}
data = iris.load_cube(CMORPH_hr)
data = subdomain_constraint(data, [28, 28.5, -16, -15])

#Extract time
print('\33[93m Extracting time period \33[00m')
months = [1,3]
data = data.extract(iris.Constraint(time=lambda t: months[0] <= t.point.month <= months[1]))

print('\33[93m Re-shaping data \33[00m')
data.data.ravel(order='F')
# data = np.mean(data.data.reshape(-1, 24), axis=1)
print('\33[93m Calculating percentiles \33[00m')

percentile = np.percentile(data.data, 99.9)

print('\33[93m Testing threshold \33[00m')
threshold = percentile
t_crossings = np.diff(data.data > threshold, prepend=False)

upward_crossing = np.argwhere(t_crossings)[::2,0]
downward_crossing = np.argwhere(t_crossings)[1::2,0]
precip_duration = downward_crossing - upward_crossing
print(f'There was a total of {len(precip_duration)} events where precipitation exceeded the'
      f' defined threshold')

plt.plot(precip_duration)



# plt.bar(precip_duration.max, precip_duration)
# plt.xticks([0,1,2,3,4])
# plt.legend()