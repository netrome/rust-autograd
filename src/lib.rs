pub mod autograd;
pub mod autograd2;


#[cfg(test)]
mod tests {
    use autograd::Variable;
    use autograd2::Shared;

    #[test]
    fn it_works() {
        assert_eq!(2 + 2, 4);
    }

    #[test]
    fn variables() {
        let a = Variable::new(2.5);
        let b = Variable::new(2.3);

        assert_eq!(2.5, a.val());
    }

    #[test]
    fn shared_variables() {
        let a = Shared::new(1.4);
        assert_eq!(1.4, a.read());

        let mut b = Shared::new(1.3);
        b.write(4.5);
        assert_eq!(b.read(), 4.5);

        let mut c = b.clone();
        c.write(2.3);
        assert_eq!(b.read(), 2.3);
    }
}
