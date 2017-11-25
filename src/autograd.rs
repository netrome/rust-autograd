use std::ops;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Variable{
    data: Arc<Mutex<VariableData>>,
}

struct VariableData{
    val: f32,
    grad: f32,
    parents: Vec<(Arc<Mutex<VariableData>>, f32)>,
}

impl Variable{
    pub fn new(val: f32) -> Self{ 
        let data = VariableData{ val: val, grad: 0.0, parents: Vec::new() };
        Variable{data: Arc::new(Mutex::new(data))}
    }

    fn from_data(data: VariableData) -> Self { Variable{ data: Arc::new(Mutex::new(data)) } }

    pub fn val(&self) -> f32{ self.data.lock().unwrap().val }
    pub fn grad(&self) -> f32{ self.data.lock().unwrap().grad }
    pub fn write(&mut self, val: f32){ self.data.lock().unwrap().val = val }
    pub fn write_grad(&mut self, val: f32){ self.data.lock().unwrap().grad = val }

    pub fn backward(&mut self){
        self.write_grad(1.0);
        self.data.lock().unwrap().chain();
    }
    pub fn zero_grad(&mut self){
        self.write_grad(0.0);
        self.data.lock().unwrap().chain_zero();
    }
}

impl VariableData{
    fn chain(&mut self){
        for parent in self.parents.iter_mut(){
            let mut var = parent.0.lock().unwrap();
            let weight = parent.1;
            var.grad += self.grad * weight;
            var.chain();
        }
    }
    fn chain_zero(&mut self){
        for parent in self.parents.iter_mut(){
            let mut var = parent.0.lock().unwrap();
            var.grad = 0.0;
            var.chain_zero();
        }
    }
}

impl ops::Add for Variable{
    type Output = Variable;
    fn add(self, other: Variable) -> Variable{
        let first = (self.data.clone(), 1.0);
        let second = (other.data.clone(), 1.0);
        let data = VariableData{ val: self.val() + other.val(), grad: 0.0, parents: vec![first, second] };
        Variable::from_data(data)
    }
}

impl ops::Add<f32> for Variable{
    type Output = Variable;
    fn add(self, other: f32) -> Variable{
        let first = (self.data.clone(), 1.0);
        let data = VariableData{ val: self.val() + other, grad: 0.0, parents: vec![first] };
        Variable::from_data(data)
    }
}

impl ops::Mul for Variable{
    type Output = Variable;
    fn mul(self, other: Variable) -> Variable{
        let first = (self.data.clone(), other.val());
        let second = (other.data.clone(), self.val());
        let data = VariableData{ val: self.val() * other.val(), grad: 0.0, parents: vec![first, second] };
        Variable::from_data(data)
    }
}

impl ops::Mul<f32> for Variable{
    type Output = Variable;
    fn mul(self, other: f32) -> Variable{
        let first = (self.data.clone(), other);
        let data = VariableData{ val: self.val() * other, grad: 0.0, parents: vec![first] };
        Variable::from_data(data)
    }
}




