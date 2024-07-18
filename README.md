# Reconhecimento-IA

![Python](https://img.shields.io/badge/Python-yellow?style=flat&logo=Python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/Tensorflow-orange?style=flat&logo=Tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-red?style=flat&logo=Keras&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-green?style=flat&logo=Flask&logoColor=white)
![Numpy](https://img.shields.io/badge/Numpy-blue?style=flat&logo=Numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-darkblue?style=flat&logo=Matplotlib)

Este projeto demonstra a criação, treinamento e implementação de um modelo de inteligência artificial utilizando Python, TensorFlow e Keras. O projeto inclui um script principal (main.py) que cria e treina o modelo, e um script de endpoint (endpoint.py) que cria um endpoint Flask para receber imagens e retornar as previsões do modelo.

![Imagem de teste do uso do modelo da inteligência artificial utilizando um dataset de 2.000 imagens]()
Imagem de um teste de validação utilizando um dataset com 2.000 imagens.

## Como instalar

1. Caso você tenha o git instalado, execute este trecho no prompt de comando dentro do diretório desejado:

```bash
git clone https://github.com/Genialogic/Reconhecimento-IA.git
```

2. Entre dentro da pasta do projeto:

```bash
cd ia-recognition
```

3. Por fim, execute o arquivo python desejado:

```bash
py main.py
```

ou

```bash
py endpoint.py
```

## Como treinar um modelo

Para você treinar e criar o seu próprio modelo de reconhecimento de imagem, você deve seguir alguns passos e adicionar as imagens para o estudo do modelo.

1. Treinamento (Train)
   - Este diretório contém as pastas de classes de aprendizado do modelo, junto com as imagens para treino.
   - Acesse o diretório /dataset/train/_/_.jpg e adicione os arquivos de imagem para que o modelo possa analisá-los e aprender.
   - Exemplo:

```
/dataset/train/cats/cat01.jpg
/dataset/train/dogs/dog01.jpg
```

2. Validação (Validation)
   - Este diretório é usado para validar a precisão do modelo com dados que ele ainda não viu durante o treinamento.
   - Acesse o diretório /dataset/validation/_/_.jpg e adicione as imagens que serão usadas para validar o modelo.
   - Exemplo:

```
/dataset/validation/cats/cat01.jpg
/dataset/validation/dogs/dog01.jpg
```

### Estrutura das pastas de Treinamento e Validação:

```
ia-recognition/
├── models/
├── specific-images/
├── uploads/
├── dataset/
│   ├── train/
│   │   ├── cats/
│   │   │   ├── cat01.jpg
│   │   │   ├── cat02.jpg
│   │   └── dogs/
│   │       ├── dog01.jpg
│   │       ├── dog02.jpg
│   ├── validation/
│       ├── cats/
│       │   ├── cat01.jpg
│       │   ├── cat02.jpg
│       └── dogs/
│           ├── dog01.jpg
│           ├── dog02.jpg
├── main.py
├── endpoint.py
└── README.md
```

Com esta estrutura, o modelo será capaz de diferenciar entre as classes de imagens fornecidas, como exemplo "cats" e "dogs", durante o treinamento e validação.

> [!NOTE]
> Após o treinamento, o modelo é salvo automaticamente no formato `.keras` dentro do diretório `/models`.

## Utilização do Modelo para Identificação de Imagens

### Utilização via Script `main.py`

1. Execute o script `main.py`:

```bash
py main.py
```

2. Selecione a primeira opção quando solicitado pelo código para realizar previsões nas imagens.
3. O script irá gerar um grid de 3x3 com as previsões das imagens presentes na pasta `/specific_images`.

### Utilização via Endpoint `endpoint.py`

1. Execute o script `endpoint.py`:

```bash
py endpoint.py
```

2. Envie uma requisição `POST` para o endpoint `/recognize` com a imagem que deseja identificar.
   - Passe o arquivo de imagem com o nome `file` no corpo da requisição.
3. O endpoint retornará uma resposta JSON com a previsão, por exemplo:

```
{
  message: "Gato"
}
```

4. A imagem enviada será salva temporariamente na pasta `/uploads`, processada pelo modelo, e depois excluída para não consumir espaço em disco.

Com esta configuração, você pode facilmente treinar, validar e usar seu modelo de reconhecimento de imagens tanto localmente via script quanto remotamente via um serviço web.
