# Automatic differentiation in Rust.
Currently only supplorting f32 values. Built for personal use, but feel free to include it in any project. This implementation will probably not be fast enough for production but could be used for gradient checks in other implementations. 

I have succesfully trained a simple fashion-MNIST classifier usuing this library. However the magnitudes of the gradients I get are unreasonable high which indicates that there probably is some bug lurking in the shadows of this library.
