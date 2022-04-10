
function(escape_string_as_regex _str_out _str_in)
  STRING(REGEX REPLACE "\\\\" "\\\\\\\\" FILETEST2 "${_str_in}")
  STRING(REGEX REPLACE "([.$+*?|-])" "\\\\\\1" FILETEST2 "${FILETEST2}")
  STRING(REGEX REPLACE "\\^" "\\\\^" FILETEST2 "${FILETEST2}")
  STRING(REGEX REPLACE "\\(" "\\\\(" FILETEST2 "${FILETEST2}")
  STRING(REGEX REPLACE "\\)" "\\\\)" FILETEST2 "${FILETEST2}")
  STRING(REGEX REPLACE "\\[" "\\\\[" FILETEST2 "${FILETEST2}")
  STRING(REGEX REPLACE "\\]" "\\\\]" FILETEST2 "${FILETEST2}")
  SET(${_str_out} "${FILETEST2}" PARENT_SCOPE)
endfunction()

function(test_escape_string_as_regex)
  SET(test1 "\\.^$-+*()[]?|")
  escape_string_as_regex(test2 "${test1}")
  SET(testRef "\\\\\\.\\^\\$\\-\\+\\*\\(\\)\\[\\]\\?\\|")
  if(NOT test2 STREQUAL testRef)
	message("Error in the escape_string_for_regex function : \n   ${test1} was escaped as ${test2}, should be ${testRef}")
  endif(NOT test2 STREQUAL testRef)
endfunction()