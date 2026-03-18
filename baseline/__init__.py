from baseline.abstract.factory import ModelRegistry
from baseline.cbramod.cbramod_adapter import CBraModDataLoaderFactory
from baseline.cbramod.cbramod_config import CBraModConfig
from baseline.cbramod.cbramod_trainer import CBraModTrainer
# from baseline.conformer.conformer_config import ConformerConfig
# from baseline.conformer.conformer_trainer import ConformerTrainer
from baseline.csbrain.csbrain_adapter import CSBrainDataLoaderFactory
from baseline.csbrain.csbrain_config import CSBrainConfig
from baseline.csbrain.csbrain_trainer import CSBrainTrainer
# from baseline.eegnet.eegnet_config import EegNetConfig
# from baseline.eegnet.eegnet_trainer import EegNetTrainer
from baseline.eegpt.eegpt_adapter import EegptDataLoaderFactory
from baseline.eegpt.eegpt_config import EegptConfig
from baseline.eegpt.eegpt_trainer import EegptTrainer
from baseline.labram.labram_adapter import LabramDataLoaderFactory
from baseline.labram.labram_config import LabramConfig
from baseline.labram.labram_trainer import LabramTrainer
from baseline.bendr.bendr_config import BendrConfig
from baseline.bendr.bendr_trainer import BendrTrainer
from baseline.biot.biot_config import BiotConfig
from baseline.biot.biot_trainer import BiotTrainer
from baseline.mantis import MantisConfig, MantisDataLoaderFactory, MantisTrainer
from baseline.moment import MomentConfig, MomentDataLoaderFactory, MomentTrainer
from baseline.reve.reve_adapter import ReveDataLoaderFactory
from baseline.reve.reve_config import ReveConfig
from baseline.reve.reve_trainer import ReveTrainer
from baseline.zuna.zuna_config import ZunaConfig
from baseline.zuna.zuna_trainer import ZunaTrainer, ZunaDataLoaderFactory

ModelRegistry.register_model(
    model_type='eegpt',
    config_class=EegptConfig,
    adapter_class=EegptDataLoaderFactory,
    trainer_class=EegptTrainer
)

ModelRegistry.register_model(
    model_type='labram',
    config_class=LabramConfig,
    adapter_class=LabramDataLoaderFactory,
    trainer_class=LabramTrainer
)

ModelRegistry.register_model(
    model_type='bendr',
    config_class=BendrConfig,
    adapter_class=None,
    trainer_class=BendrTrainer
)

ModelRegistry.register_model(
    model_type='biot',
    config_class=BiotConfig,
    adapter_class=None,
    trainer_class=BiotTrainer
)

ModelRegistry.register_model(
    model_type='cbramod',
    config_class=CBraModConfig,
    adapter_class=CBraModDataLoaderFactory,
    trainer_class=CBraModTrainer
)

ModelRegistry.register_model(
    model_type='reve',
    config_class=ReveConfig,
    adapter_class=ReveDataLoaderFactory,
    trainer_class=ReveTrainer
)

ModelRegistry.register_model(
    model_type='csbrain',
    config_class=CSBrainConfig,
    adapter_class=CSBrainDataLoaderFactory,
    trainer_class=CSBrainTrainer
)

# ModelRegistry.register_model(
#     model_type='eegnet',
#     config_class=EegNetConfig,
#     adapter_class=None,
#     trainer_class=EegNetTrainer
# )

# ModelRegistry.register_model(
#     model_type='conformer',
#     config_class=ConformerConfig,
#     adapter_class=None,
#     trainer_class=ConformerTrainer
# )

ModelRegistry.register_model(
    model_type='mantis',
    config_class=MantisConfig,
    adapter_class=MantisDataLoaderFactory,
    trainer_class=MantisTrainer
)

ModelRegistry.register_model(
    model_type='moment',
    config_class=MomentConfig,
    adapter_class=MomentDataLoaderFactory,
    trainer_class=MomentTrainer
)

ModelRegistry.register_model(
    model_type='zuna',
    config_class=ZunaConfig,
    adapter_class=ZunaDataLoaderFactory,
    trainer_class=ZunaTrainer
)
