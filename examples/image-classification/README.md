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
JOB_NAME="test`date '+%Y%m%d%H%M%S'`"
PROJECT_ID=`gcloud config list project --format "value(core.project)"`

gcloud ml-engine jobs submit training ${JOB_NAME} \
  --package-path=trainer \
  --module-name=trainer.task \
  --staging-bucket="gs://${PROJECT_ID}-ml" \
  --region=us-central1 \
  --config=mlengine.yaml \
  -- \
  --train-csv=${TRAIN_CSV}
```
