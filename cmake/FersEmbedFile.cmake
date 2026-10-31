if (NOT DEFINED INPUT)
	message(FATAL_ERROR "FersEmbedFile requires INPUT")
endif ()

if (NOT DEFINED OUTPUT)
	message(FATAL_ERROR "FersEmbedFile requires OUTPUT")
endif ()

if (NOT DEFINED VAR)
	message(FATAL_ERROR "FersEmbedFile requires VAR")
endif ()

file(READ "${INPUT}" _fers_embed_hex HEX)
string(LENGTH "${_fers_embed_hex}" _fers_embed_hex_length)
math(EXPR _fers_embed_byte_count "${_fers_embed_hex_length} / 2")

# Convert the string to a C-style array of byte values.
set(_fers_embed_body "${_fers_embed_hex}")
if (_fers_embed_byte_count GREATER 0)
	string(REGEX REPLACE "([0-9a-f][0-9a-f])" "0x\\1, " _fers_embed_body "${_fers_embed_body}")
endif ()

file(WRITE "${OUTPUT}"
	 "unsigned char ${VAR}[] = {${_fers_embed_body}};\n"
	 "unsigned int ${VAR}_len = ${_fers_embed_byte_count};\n"
)
