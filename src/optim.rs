use autograd::Variable;

pub trait Optimizer{
    fn update_variables(&self);
}

pub struct SGD{
    rate: f32,
    variables: Vec<Variable>,
}

impl Optimizer for SGD{
    fn update_variables(&self){
        for var in self.variables.iter(){
            let new_val = var.val() - var.grad() * self.rate;
            var.write(new_val);
        }
    }
}
