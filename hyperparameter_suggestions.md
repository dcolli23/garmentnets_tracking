- Standard attempts:
    - Train for longer
    - Add batch norm
    - dropout
    - reduce number of units
    - reduce number of layers
- Adam parameters
    - learning rate may be too high
    - Increase L2 regularization
- Try increasing number of sampled points for dynamics learning to 9000 points.
- Point matching improvements
    - Update positions of non-corresponding points from the full view based on average transformation of corresponding points.