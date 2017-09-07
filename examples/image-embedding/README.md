# Embedding Visualization on TensorBoard

## Embed images

```python
import tfmodel

input_exps=[
    "examples/image-embedding/img/yasuna/*.png",
    "examples/image-embedding/img/sonya/*.png",
]

tfmodel.util.embed(
    input_exps=input_exps,
    output_dir="embeddings"
)
```

## Result

<img src="result.png" width=500>

## Resources

* http://killmebaby.tv/special_icon.html
