# Deep Learning Project: Aiding Dog Breed Classification with Vision Transformers
MIT 15.S04 Hands-On Deep Learning

## Problem Statement
According to Cvetkovska (2020), animal shelters cost US citizens $2 billion annually. From a business perspective, automating the identification process saves time and resources spent on manual identification and documentation, allowing staff to concentrate on other essential tasks like animal care and customer service. Furthermore, better matching between potential adopters and dogs becomes possible. This is because the exact breed information enables centres to provide better insights into each dog's temperament, size, and care requirements.

## Methodology and Approach
Data preprocessing was first conducted to standardise image sizes to a fixed (height, width) dimension to ensure a fair comparison across models. The depth dimension was kept as 3 (for RGB). Then, centre cropping was applied to remove artefacts around the dog (e.g., the human owner, other animals that may not be the dog of interest, etc.) that may mislead the model to learn spurious features. Finally, a train-test split was applied in the ratio of 80-20.

Four models were developed or fine-tuned. These were: a Convolutional Neural Network (CNN), VGG19, ResNet50, and a Vision Transformer. My role was to fine-tune the vision transformer using Object Oriented Programming (OOP) and PyTorch. The full code can be found within the notebook: *01-vision-transformer*.

## Results and Discussion
<table>
  <thead>
    <tr>
      <th rowspan="2" style="text-align:center">MODEL</th>
      <th colspan="2" style="text-align:center">ACCURACY</th>
    </tr>
    <tr>
      <th style="text-align:center">TRAIN</th>
      <th style="text-align:center">TEST</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align:center">Baseline (Random Guess)</td>
      <td colspan="2" style="text-align:center">0.8%</td>
    </tr>
    <tr>
      <td style="text-align:center">Vanilla CNN</td>
      <td style="text-align:center">20.9%</td>
      <td style="text-align:center">7.1%</td>
    </tr>
    <tr>
      <td style="text-align:center">ResNet50</td>
      <td style="text-align:center">92.4%</td>
      <td style="text-align:center">24.0%</td>
    </tr>
    <tr>
      <td style="text-align:center">VGG19</td>
      <td style="text-align:center">94.6%</td>
      <td style="text-align:center">60.9%</td>
    </tr>
    <tr>
      <td style="text-align:center">Vision Transformer</td>
      <td style="text-align:center">98.4%</td>
      <td style="text-align:center">90.7%</td>
    </tr>
  </tbody>
</table>

VGG19 outperformed ResNet50 due to the treatment of embeddings. In ResNet50, after the final convolutional layer, flattening the embedding resulted in the loss of spatial information as multi-dimensional feature maps were compressed into a one-dimensional vector. Conversely, VGG19 utilised Global Average Pooling, which preserved spatial relationships and retained important information by summarising each feature map through the average of its values. By effectively capturing global information while avoiding the loss of spatial details incurred by flattening, VGG19 was able to better represent the intricate relationships between different parts of an image, thereby achieving superior performance.

The Transformer outperformed everything. By dividing the image into patches and selectively attending to specific patches, the Transformer can prioritise recognising objects such as dogs over other elements like humans (as the preprocessing was not perfect). This approach optimises its object recognition capabilities, leading to superior performance.

## Report
[15.S04 Final Report](./deep-learning-final-report.pdf)

## Presentation
[15.S04 Final Presentation](./deep-learning-ppt.pptx)

## Notebooks
- *01-vision-transformer.ipynb*: Fine-tuned vision transformer for dog breed classification
