Now we've seen that the error in the output layer is

δ​k​​=(y​k​​−​y​^​​​k​​)f​′​​(a​k​​)

and the error in the hidden layer is

https://d17h27t6h515a5.cloudfront.net/topher/2017/January/588bc453_hidden-errors/hidden-errors.gif

For now we'll only consider a simple network with one hidden layer and one output unit. Here's the general algorithm for updating the weights with backpropagation:

  Set the weight steps for each layer to zero
    The input to hidden weights Δw​ij​​=0
    The hidden to output weights ΔW​j​​=0
  For each record in the training data:
    Make a forward pass through the network, calculating the output ​y​^​​
    Calculate the error gradient in the output unit, δ​o​​=(y−​y​^​​)f​′​​(a) where z is the input to the output unit.
    Propagate the errors to the hidden layer δ​j​h​​=δ​o​​W​j​​f​′​​(h​j​​)
    Update the weight steps,:
      ΔW​j​​=ΔW​j​​+δ​o​​a​j​​
      Δw​ij​​=Δw​ij​​+δ​j​h​​a​i​​
    Update the weights, where η is the learning rate and m is the number of records:
      W​j​​=W​j​​+ηΔW​j​​/m
      w​ij​​=w​ij​​+ηΔw​ij​​/m
    Repeat for e epochs.
