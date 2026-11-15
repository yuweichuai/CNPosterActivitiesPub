* Run this file once in a licensed Stata installation if commands are missing.
* This setup step may download packages; the analysis itself requires no APIs.
foreach command in ftools reghdfe ppmlhdfe sdid honestdid coefplot {
    capture which `command'
    if _rc {
        ssc install `command'
    }
    which `command'
}
* Tested versions are recorded in README.md. Existing installations are not replaced.
