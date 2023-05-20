
# Enhanced Word-Context Character Embeddings for Chinese word Segmentation

This is the modified source code which was initially developed in the paper "Word-Based Character Embeddings for Chinese word Segmentation".

## Project Components
Our model comprises 2 components:
* The baseline segmentation model located in the "segmenter" directory.
* The training code for word-based character embeddings found in the "modified-word2vec" directory.

## Prerequisite Software
 * CMake
 * Boost
 * glog

## Embedding Model Training


	#Run
	make


	#Training command

	./seg2vec -train data/small.seg.char.giga \
    	-output  vecs/example.emb \
    	-cbow 0  \
    	-size 50 \
    	-window 5 \
    	-negative 10 \
    	-sample 1e-4 \
    	-threads 6 \
    	-binary 0 \
    	-iter 8 \


## Segmentation Model Operation

	./configure
    make

	mkdir -p model
	mkdir -p log

	GLOG_log_dir=log ./greedy --cnn-mem 999 -i 1 \
    	-T ../../data/pku/small.train.seg \
    	-d ../../data/pku/small.dev.seg \
    	-t ../../data/pku/small.test.seg \
    	--optimizer simple_sgd \
    	--evaluate_stops 2500 \
    	--outfile ctb.test.res \

## Segmentation Model Parameters

	./greedy -h
        
## Please Note:  
* Sample training data can be found in the data directory.
* The performances of crf and greedy models are quite similar.

------

Reference:
Hao Zhou, Zhenting Yu, Yue Zhang, Shujian Huang, Xinyu Dai and Jiajun Chen. Word-Context Character Embeddings for Chinese word Segmentation. In Proceeding of EMNLP 2017, short paper.