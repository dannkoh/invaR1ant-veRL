(declare-const put0_0 Int)
(declare-const put0_1 Int)
(declare-const get0 Int)
(declare-const get1 Int)

(assert (and (and  ( =  0 ( mod  ( +  put0_1 put0_0) 1))  ( =  0 ( mod  ( +  get1 get0) 1))) (not ( = put0_0 get0))))

(check-sat)
(get-model)