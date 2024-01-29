### torchserve 배포

1. torchserve 배포
```
torch-model-archiver --model-name gpt_caption --version 1.0 --serialized-file weights/large/model.pt --extra-files src/model/model.py,src/utils --handler src/my_handler.py --requirements-file requirements.txt --export-path d:/10.python/serve/model_store
```
2. torchserve 실행
```
torchserve --start --model-store model_store --models all
```
3. torchserve api 확인
```
http://localhost:8081/models/gpt_caption
```
4. gpt_caption test
```
curl -X POST http://127.0.0.1:8080/predictions/gpt_caption -T images.jpg
```
