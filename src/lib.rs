pub mod autograd;


#[cfg(test)]
mod tests {

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    use autograd::Variable;
    #[test]
    fn check_reference_ops() {
        let a = Variable::new(3.);
        let b = Variable::new(2.);

        let mut c = &(&(&a + &b) + 3.) * &(&a + &b);
        c.backward();
        assert_eq!(c.val(), 40.);
        assert_eq!(a.grad(), 13.);
    }
}






