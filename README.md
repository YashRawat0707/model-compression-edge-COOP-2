[Model Compression for Edge Devices]

[Overview]

This project explores techniques for optimizing deep learning models for deployment on resource-constrained edge devices. The focus is on reducing model size while maintaining acceptable accuracy.

[Techniques Implemented]

* Baseline CNN model trained on the CIFAR-10 dataset
* Pruning
* Quantization
* Hybrid approach combining pruning and quantization

[Results]

| Model     | Size (MB) | Accuracy (%) |
| --------- | --------- | ------------ |
| Original  | 2.34      | 69.95        |
| Pruned    | 2.34      | 67.02        |
| Quantized | 0.64      | 69.92        |
| Hybrid    | 0.60      | 66.50        |

[Observations]

* Quantization significantly reduces model size with minimal loss in accuracy.
* Pruning reduces parameter redundancy but does not significantly affect storage size.
* The hybrid approach provides additional compression with a slight drop in accuracy.

[Implementation]

The project includes scripts for:

* Training the baseline model
* Applying pruning, quantization, and hybrid compression
* Evaluating model performance
* Visualizing results using a Streamlit-based interface

[Running the Project]

Install dependencies:
pip install torch torchvision streamlit matplotlib

Run the application:
python -m streamlit run app.py

[Project Structure]

* train_model.py – baseline model training
* compress_model.py – pruning implementation
* quantize_model.py – quantization implementation
* hybrid_model.py – combined compression
* evaluate files – performance evaluation
* app.py – visualization interface

[Conclusion]

The results demonstrate that model compression techniques can significantly reduce storage requirements while maintaining performance, making them suitable for edge AI deployment.

