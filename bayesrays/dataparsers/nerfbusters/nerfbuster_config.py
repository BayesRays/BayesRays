''' This file only serves the purpose of loading nerfbuster dataset for bayesrays and does not work for running nerfbuster mode. To run/test nerfbuster model please refer to nerfbuster repository '''

from nerfstudio.plugins.registry_dataparser import DataParserSpecification
from bayesrays.dataparsers.nerfbusters.nb_dataparser import NerfbusterDataparserConfig

nbDataparser = DataParserSpecification(config=NerfbusterDataparserConfig())