use std::ops;
use std::fmt;
use std::iter;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
pub struct Variable{
    data: Arc<Mutex<VariableData>>,
}

#[derive(Clone)]
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

    pub fn from_vec(vec: Vec<f32>) -> Vec<Variable>{
        vec.iter().map(|i| Self::new(*i)).collect()
    }

    pub fn val(&self) -> f32{ self.data.lock().unwrap().val }
    pub fn grad(&self) -> f32{ self.data.lock().unwrap().grad }
    pub fn write(&self, val: f32){ self.data.lock().unwrap().val = val }
    pub fn write_grad(&self, val: f32){ self.data.lock().unwrap().grad = val }

    pub fn backward(&self){
        self.write_grad(1.0);
        self.data.lock().unwrap().chain();
    }
    pub fn zero_grad(&self){
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

// Print related stuff -------------------------------------------------------------------------
impl fmt::Display for Variable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.val())
    }
}

impl fmt::Debug for Variable {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.val())
    }
}

// Addition with variable and scalar  -------------------------------------------------------------------------
impl<'a> ops::Add for &'a Variable{
    type Output = Variable;
    fn add(self, other: &'a Variable) -> Variable{
        let first = (self.data.clone(), 1.0);
        let second = (other.data.clone(), 1.0);
        let data = VariableData{ val: self.val() + other.val(), grad: 0.0, parents: vec![first, second] };
        Variable::from_data(data)
    }
}

impl<'a> ops::Add<f32> for &'a Variable{
    type Output = Variable;
    fn add(self, other: f32) -> Variable{
        let first = (self.data.clone(), 1.0);
        let data = VariableData{ val: self.val() + other, grad: 0.0, parents: vec![first] };
        Variable::from_data(data)
    }
}

// Subtraction with variable and scalar -----------------------------------------------------------------------------
impl<'a> ops::Sub for &'a Variable{
    type Output = Variable;
    fn sub(self, other: &'a Variable) -> Variable{
        let first = (self.data.clone(), 1.0);
        let second = (other.data.clone(), -1.0);
        let data = VariableData{ val: self.val() - other.val(), grad: 0.0, parents: vec![first, second] };
        Variable::from_data(data)
    }
}

impl<'a> ops::Sub<f32> for &'a Variable{
    type Output = Variable;
    fn sub(self, other: f32) -> Variable{
        let first = (self.data.clone(), 1.0);
        let data = VariableData{ val: self.val() - other, grad: 0.0, parents: vec![first] };
        Variable::from_data(data)
    }
}

impl<'a> ops::Sub<&'a Variable> for f32{
    type Output = Variable;
    fn sub(self, other: &'a Variable) -> Variable{
        let first = (other.data.clone(), -1.0);
        let data = VariableData{ val: self - other.val(), grad: 0.0, parents: vec![first] };
        Variable::from_data(data)
    }
}

// Multiplication with variable and scalar  -------------------------------------------------------------------------
impl<'a> ops::Mul for &'a Variable{
    type Output = Variable;
    fn mul(self, other: &'a Variable) -> Variable{
        let first = (self.data.clone(), other.val());
        let second = (other.data.clone(), self.val());
        let data = VariableData{ val: self.val() * other.val(), grad: 0.0, parents: vec![first, second] };
        Variable::from_data(data)
    }
}

impl<'a> ops::Mul<f32> for &'a Variable{
    type Output = Variable;
    fn mul(self, other: f32) -> Variable{
        let first = (self.data.clone(), other);
        let data = VariableData{ val: self.val() * other, grad: 0.0, parents: vec![first] };
        Variable::from_data(data)
    }
}

// Division with variable and scalar -----------------------------------------------------------------------------
impl<'a> ops::Div for &'a Variable{
    type Output = Variable;
    fn div(self, other: &'a Variable) -> Variable{
        let first = (self.data.clone(), 1.0/other.val());
        let second = (other.data.clone(), -self.val()/(other.val()*other.val()));
        let data = VariableData{ val: self.val() / other.val(), grad: 0.0, parents: vec![first, second] };
        Variable::from_data(data)
    }
}

impl<'a> ops::Div<f32> for &'a Variable{
    type Output = Variable;
    fn div(self, other: f32) -> Variable{
        let first = (self.data.clone(), 1.0/other);
        let data = VariableData{ val: self.val() / other, grad: 0.0, parents: vec![first] };
        Variable::from_data(data)
    }
}

impl<'a> ops::Div<&'a Variable> for f32{
    type Output = Variable;
    fn div(self, other: &'a Variable) -> Variable{
        let first = (other.data.clone(), -1./(other.val() * other.val()));
        let data = VariableData{ val: self / other.val(), grad: 0.0, parents: vec![first] };
        Variable::from_data(data)
    }
}

// For iterators ------------------------------------------------
impl<'a> iter::Sum<&'a Variable> for Variable{
    fn sum<I: iter::Iterator<Item=&'a Variable>>(iter: I) -> Self{
        let mut sum = Variable::new(0.0);
        for var in iter{ sum = &sum + var; }
        sum
    }
}


