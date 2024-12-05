public String toString() {return pattern();}
public boolean contains(Object o) {return indexOf(o) != -1;}
public CellRangeAddress getCellRangeAddress(int index) {return _list.get(index);}
public ObjectId getResultTreeId() {return (resultTree == null) ? null : resultTree.toObjectId();}
public boolean equals(final Object o){boolean rval = this == o;if (!rval && (o != null) && (o.getClass() == this.getClass())){IntList other = ( IntList ) o;if (other._limit == _limit){rval = true;for (int j = 0; rval && (j < _limit); j++){rval = _array[ j ] == other._array[ j ];}}}return rval;}
public QueryPhraseMap searchPhrase( String fieldName, final List<TermInfo> phraseCandidate ){QueryPhraseMap root = getRootMap( fieldName );if( root == null ) return null;return root.searchPhrase( phraseCandidate );}
public E push(E object) {addElement(object);return object;}
public ReplicationGroup testFailover(TestFailoverRequest request) {request = beforeClientExecution(request);return executeTestFailover(request);}
@Override public boolean remove(Object o) {if (contains(o)) {Entry<?> entry = (Entry<?>) o;AtomicInteger frequency = backingMap.remove(entry.getElement());int numberRemoved = frequency.getAndSet(0);size -= numberRemoved;return true;}return false;}
public DescribeStackDriftDetectionStatusResult describeStackDriftDetectionStatus(DescribeStackDriftDetectionStatusRequest request) {request = beforeClientExecution(request);return executeDescribeStackDriftDetectionStatus(request);}
public ByteBuffer put(byte b) {throw new ReadOnlyBufferException();}
public String toString(String field) {StringBuilder buffer = new StringBuilder();buffer.append("spanFirst(");buffer.append(match.toString(field));buffer.append(", ");buffer.append(end);buffer.append(")");return buffer.toString();}
public int fillFields(byte[] data, int offset, EscherRecordFactory recordFactory) { readHeader( data, offset );int pos            = offset + 8;int size           = 0;field_1_shapeId    =  LittleEndian.getInt( data, pos + size );     size += 4;field_2_flags      =  LittleEndian.getInt( data, pos + size );     size += 4;return getRecordSize();}
public void setPackedGitMMAP(boolean usemmap) {packedGitMMAP = usemmap;}
public void serialize(LittleEndianOutput out) {out.writeShort(_extBookIndex);out.writeShort(_firstSheetIndex);out.writeShort(_lastSheetIndex);}
public final String[] getValues(String name) {List<String> result = new ArrayList<>();for (IndexableField field : fields) {if (field.name().equals(name) && field.stringValue() != null) {result.add(field.stringValue());}}if (result.size() == 0) {return NO_STRINGS;}return result.toArray(new String[result.size()]);}
public void serialize(LittleEndianOutput out) {futureHeader.serialize(out);out.writeShort(isf_sharedFeatureType);out.writeByte(reserved);out.writeInt((int)cbHdrData);out.write(rgbHdrData);}
public StringBuffer append(long l) {IntegralToString.appendLong(this, l);return this;}
public void serialize(LittleEndianOutput out) {out.writeShort(field_1_option_flag);out.writeShort(field_2_ixals);out.writeShort(field_3_not_used);out.writeByte(field_4_name.length());StringUtil.writeUnicodeStringFlagAndData(out, field_4_name);if(!isOLELink() && !isStdDocumentNameIdentifier()){if(isAutomaticLink()){if(_ddeValues != null) {out.writeByte(_nColumns-1);out.writeShort(_nRows-1);ConstantValueParser.encode(out, _ddeValues);}} else {field_5_name_definition.serialize(out);}}}
public IgnoreNode(List<FastIgnoreRule> rules) {this.rules = rules;}
public void flush() throws IOException {try {beginWrite();dst.flush();} catch (InterruptedIOException e) {throw writeTimedOut(e);} finally {endWrite();}}
public Hyphenation hyphenate(String word, int remainCharCount,int pushCharCount) {char[] w = word.toCharArray();return hyphenate(w, 0, w.length, remainCharCount, pushCharCount);}
public ContinueRecord(RecordInputStream in) {_data = in.readRemainder();}
public void close() {if (sock != null) {try {sch.releaseSession(sock);} finally {sock = null;}}}
public Set<String> getNames(String section, String subsection) {return getState().getNames(section, subsection);}
public ValueEval evaluate(ValueEval[] args, OperationEvaluationContext ec) {throw new NotImplementedFunctionException(_functionName);}
public final CoderResult flush(CharBuffer out) {if (status != END && status != INIT) {throw new IllegalStateException();}CoderResult result = implFlush(out);if (result == CoderResult.UNDERFLOW) {status = FLUSH;}return result;}
public TokenStream init(TokenStream tokenStream) {termAtt = tokenStream.addAttribute(CharTermAttribute.class);return null;}
public int getNumberOfOnChannelTokens() {int n = 0;fill();for (int i = 0; i < tokens.size(); i++) {Token t = tokens.get(i);if ( t.getChannel()==channel ) n++;if ( t.getType()==Token.EOF ) break;}return n;}
public byte[] getCachedBytes() {return data;}
public String getEncoding() {if (encoder == null) {return null;}return HistoricalCharsetNames.get(encoder.charset());}
public String toString() {StringBuilder sb = new StringBuilder();sb.append('[').append("USERSVIEWEND").append("] (0x");sb.append(Integer.toHexString(sid).toUpperCase(Locale.ROOT)).append(")\n");sb.append("  rawData=").append(HexDump.toHex(_rawData)).append("\n");sb.append("[/").append("USERSVIEWEND").append("]\n");return sb.toString();}
public void remove() {if (lastReturned == null)throw new IllegalStateException();ConcurrentHashMap.this.remove(lastReturned.key);lastReturned = null;}
public int lastIndexOf(Object object) {if (object != null) {for (int i = a.length - 1; i >= 0; i--) {if (object.equals(a[i])) {return i;}}} else {for (int i = a.length - 1; i >= 0; i--) {if (a[i] == null) {return i;}}}return -1;}
