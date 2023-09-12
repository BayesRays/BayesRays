''' Loading a few forward-facing views of the whole dataset as train and the rest as test data for bayesrays '''
# view picks for LF based on CF-NeRF and for ScanNet based on NerfingMVS codebases.

from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from bayesrays.dataparsers.sparse.sparse_nerfstudio_dataparser import SparseNsDataParserConfig

sparseNsDataparser = DataParserSpecification(config=SparseNsDataParserConfig())