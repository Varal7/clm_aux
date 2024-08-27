# Aux repo for Conformal Language Modeling

Aux repo for our [paper](https://arxiv.org/abs/2306.10193)

Also see our [main repo](https://github.com/Varal7/conformal-language-modeling)

## Reproduce our work

For example, for TriviaQA:

1. Download the Llama weights from https://huggingface.co/meta-llama/Llama-2-7b-hf
2. From our clm_aux repository (https://github.com/Varal7/clm_aux/blob/main/scripts/qa), run the following scripts:  
 (a) run_triviaqa.py, using the --checkpoint argument to define the location of the model weights downloaded in step 1  
 (b) fix_eos.py, using the output of (a) as input to (b)  
 (c) run_triviaqa_self_eval.py, using the output of (b) as input to (c), don't forget to specify the model location again  
 (d) compute_qa_scores.py, specifiy the desired output using the command line flags  
3. In our main repository (https://github.com/Varal7/conformal-language-modeling/blob/main/scripts), run the following scripts:  
 (a) triviaqa_data.py, which will split the data into train/val/test splits  
 (b) run_triviaqa.sh, (adapt to your needs) which will apply our algorithm to the data created above.  
4. You can analyse the produced outputs (prob_results.npz) with our analysis notebooks, e.g https://github.com/Varal7/conformal-language-modeling/blob/main/notebooks/plot.ipynb

Alternatively, intermediate outputs of step 2 are available at `outputs/uncertainty`.

## Citation

If you use our work, please cite

```
@misc{quach2023conformal,
      title={Conformal Language Modeling}, 
      author={Victor Quach and Adam Fisch and Tal Schuster and Adam Yala and Jae Ho Sohn and Tommi S. Jaakkola and Regina Barzilay},
      year={2023},
      eprint={2306.10193},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
