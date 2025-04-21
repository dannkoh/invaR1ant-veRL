(declare-const in6 Int)
(declare-const in9 Int)
(declare-const in3 Int)

(assert (and (and  ( =  in3 0)  ( =  in6 0))  ( =  in9 0)))

(check-sat)
(get-model)