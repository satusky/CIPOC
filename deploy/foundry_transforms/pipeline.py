from transforms.api import Pipeline

from myproject.datasets import cipoc_extract, cipoc_omop

my_pipeline = Pipeline()
my_pipeline.add_transforms(cipoc_extract.extract_cases, cipoc_omop.export_omop)
