height = 0*(1:10000)


for(i in c(1:100000)) {
   step = 2*rbinom(10000, 1, 0.5)-1.0
   height = abs(height+step)
}






