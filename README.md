
# Real_Time-Machine-Learning
**Learning Feature Trees** offer a new approach to machine learning with a few big surprises for the machine learning community:
1. How it does pattern recognition is **easily understood**, and it gets 96 percent correct on MNIST test data. With parallel hardware, training could take just a few seconds. To get higher accuracy, one can specify other learning tasks that analyse images. Little human intervention is required for a new task because feature design is **automatic**.
2. With enough parallel hardware, problems with many classes can take the **same time** as with few classes. Higher accuracy often takes **no extra time**. 

3. It uses **forward propagation** of credit assignment. Simple "engrams" identify weights to change even **much later** than an action. If a system can't remember what it did that later led to a negative reinforcement, it can't correct the previous action.

4. **The architecture grows itself as LF-trees**. Human intervention is for tasks like adapting the program to new data formats. 

5. It's **free** under the MIT license.

**HAVE FUN!!!** 

Bill Armstrong

Note: If you are not using Visual Studio, you can run the demo executable with inputs    "MNIST" 784 11 20 10 15.0 2.0    to do a short run, then change 20 to 200 and do a more accurate classification of the MNIST test set. The reinforcement routine contains some suggestions how to improva accuracy. The .pdf describes the LF-tree method in detail. 

I hope someone will make a fast CUDA version and put it on GitHub so we can all try it using a GPU. The above MNISTclassification will run somewhere between 200 and 20000 times faster.



N. B. This GitHub project is still under test. All comments, critiques etc. are welcome.