extern crate gradground;

use gradground::autograd;
use gradground::autograd::Variable;

fn main(){
    println!("Hello yello");

    autograd::hello();

    let var = Variable::new(1.2);

    println!("Value: {}", var);

}
