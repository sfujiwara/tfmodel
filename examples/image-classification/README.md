# Transfer Learning using VGG16

## CSV File for Training

```csv
# image_path,label
path/to/image/file.jpg,0
...
```

## Training on Local

```
python -m trainer.task
```

## Training on ML Engine

```
pip install -r requirements.txt -t .
```

```
JOB_NAME="hoge`date '+%Y%m%d%H%M%S'`"
PROJECT_ID=`gcloud config list project --format "value(core.project)"`
TRAIN_CSV=<path to csv file for training>

gcloud ml-engine jobs submit training ${JOB_NAME} \
  --package-path=trainer \
  --module-name=trainer.task \
  --staging-bucket="gs://${PROJECT_ID}-ml" \
  --region=us-central1 \
  --config=mlengine.yaml \
  -- \
  --train_csv=${TRAIN_CSV} \
  --output_path=<output directory for summary and model>
```
