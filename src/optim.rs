use autograd::Variable;

pub trait Optimizer{
    fn update_variables(&self);
}

pub struct SGD{
    rate: f32,
    variables: Vec<Variable>,
}

impl SGD{
    pub fn new(rate: f32, var: Vec<Variable>) -> Self{
        SGD{rate:rate, variables:var}
    }
}

impl Optimizer for SGD{
    fn update_variables(&self){
        for var in self.variables.iter(){
            let new_val = var.val() - var.grad() * self.rate;
            var.write(new_val);
            var.write_grad(0.0);
        }
    }
}
