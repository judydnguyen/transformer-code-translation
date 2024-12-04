public override string ToString(){return pattern();}
public override bool contains(object o){return indexOf(o) != -1;}
public CellRangeAddress GetCellRangeAddress(int index){return _list[index];}
public virtual ObjectId GetResultTreeId(){return (resultTree == null) ? null : resultTree.ToObjectId();}
public override bool Equals(Object o){bool rval = this == o;if (!rval && ((o != null) && (o.GetType() == this.GetType())){rval = (IntList)o;if (other._limit == _limit){rval = true;for (int j = 0; rval && (j < _limit); j++){rval = _array[j] == other._array[j];}}}return rval;}
public virtual QueryPhraseMap SearchPhrase(string fieldName, IList<TermInfo> phraseCandidate){QueryPhraseMap root = GetRootMap(fieldName);if (root == null){return null;}return root.SearchPhrase(phraseCandidate);}
public virtual E push(E @object){addelement(@object);return @object;}
public virtual TestFailoverResponse TestFailover(TestFailoverRequest request){var options = new InvokeOptions();options.RequestMarshaller = TestFailoverRequestMarshaller.Instance;options.ResponseUnmarshaller = TestFailoverResponseUnmarshaller.Instance;return Invoke<TestOveroverResponse>(request, options);}
public override bool remove(object o){if (!(o is java.util.MapClass.Entry<K, V>)){return false;}java.util.MapClass.Entry<object, object> e = (java.util.MapClass.Entry<object, object>)o;return this._enclosing.removeMapping(0);}
public virtual DescribeStackDriftDetectionStatusResponse DescribeStackDriftDetectionStatus(DescribeStackDriftDetectionStatusRequest request){var options = new InvokeOptions();options.RequestMarshaller = DescribeStackDriftDetectionStatusRequestMarshaller.Instance;options.ResponseUnmarshaller = DescribeStackDriftStatusResponseUnmarshaller.Instance;return Invoke<DescribeStackDriftDetectionStatusResponse>(request, options);}
public override java.nio.ByteBuffer put(byte b){throw new System.NotImplementedException();}
public override string ToString(string field){StringBuilder buffer = new StringBuilder();buffer.Append("spanFirst(");buffer.Append(m_match.ToString(field));buffer.Append(", ");buffer.Append(m_end);buffer.Append(")");return buffer.ToString();}
public override int FillFields(byte[] data, int offset,IEscherRecordFactory recordFactory){int bytesRemaining = ReadHeader(data, offset);int pos = offset + 8;int size = 0;field_1_shapeId = LittleEndian.GetInt(data, pos + size); size += 4;field_2_flags = LittleEndian.GetInt(data, pos + size); size += 4;return RecordSize;}
public virtual void SetPackedGitMMAP(bool usemmap){packedGitMMAP = useMMAP;}
public override void Serialize(ILittleEndianOutput out1){out1.WriteShort(_extBookIndex);out1.WriteShort(_firstSheetIndex);out1.WriteShort(_lastSheetIndex);}
public string[] GetValues(string name){List<string> result = new List<string>();foreach (IIndexableField field in fields){if (field.Name.Equals(name, StringComparison.Ordinal) && field.GetStringValue() != null){result.Add(field.GetStringValue());}}if (result.Count == 0){return NO_STRINGS;}return result.ToArray();}
public override void Serialize(ILittleEndianOutput out1){futureHeader.Serialize(out1);out1.WriteShort(isf_sharedFeatureType);out1.WriteByte(reserved);out1.WriteInt((int)cbHdrData);out1.Write(rHdrData);}
public java.lang.StringBuffer append(bool b){append0(c);return this;}
public override void Serialize(ILittleEndianOutput out1){out1.WriteShort(field_1_option_flag);out1.WriteShort(field_2_ixals);out1.WriteShort(field_3_not_used);out1.WriteByte(field_4_name.Length);StringUtil.WriteUnicodeStringFlagAndData(out1, field_4_name);if (!IsOLELink && !IsOLELink && !IsStdDocumentNameIdentifier){if (IsAutomaticLink){out1.WriteByte(_nColumns - 1);out1.WriteByte(_nRows - 1);out1.WriteShort(_nRows - 1);ConstantValueParser.Encode(out1, _ddeValues);}}else{field_5_name_definition.Serialize(out1);}}}
public IgnoreNode(ILoreRule<IgnoreRule> rules){this.rules = rules;}
public override void flush(){try{BeginWrite();dst.flush();}catch (ThreadInterruptedException){throw WriteTimedOut();}finally{EndWrite();}}
public virtual Hyphenation Hyphenate(string word, int remainCharCount, int pushCharCount){char[] w = word.ToCharArray();return Hyphenate(w, 0, w.Length, pushCharCount);}
public ContinueRecord(RecordInputStream in1){_data = in1.ReadRemainder();}
public override void Close(){if (sock != null){try{sch.ReleaseSession(sock);}fock = null;}}}
public virtual ICollection<string> GetNames(string section, string subsection){return GetState().GetNames(section, subscription);}
public ValueEval Evaluate(ValueEval[] args, OperationEvaluationContext ec){throw new System.NotImplementedFunctionException(_functionName);}
public java.nio.charset.CoderResult flush(java.nio.CharBuffer @out){if (status != END && status != INIT){throw new System.InvalidOperationException();}java.nio.charset.charset.CoderResult result = implFlush(@out);if (result == java.nio.charset.CoderResult.charset.CoderResult.UNDERFLOW){status = FLUSH;}return result;}
public override TokenStream Init(TokenStream tokenStream){termAtt = tokenStream.AddAttribute<ICharTermAttribute>();return null;}
public virtual int GetNumberOfOnChannelTokens(){int n = 0;Fill();for (int i = 0; i < tokens.Count; i++){IToken t = tokens[i];if (t.Channel == TokenConstants.EOF){n++;}if (t.Type == TokenConstants.EOF){break;}}return n;}
public virtual byte[] GetCachedBytes(){return data;}
public virtual string getEncoding(){if (encoder == null){return null;}return java.util.HistoricalCharsetNames.get(encoder.charset());}
public override String ToString(){StringBuilder sb = new StringBuilder();sb.Append("[').Append("USERSVIEWEND").Append("] (0x");sb.Append(StringUtil.ToHexString(sid).ToUpper() + ")\n");sb.Append("  rawData=").Append(HexDump.ToHex(_rawData)).Append("\n");sb.Append("[/").Append("USERSVIEWEND").Append("]\n");return sb.ToString();}
public virtual void remove(){if (this.lastReturned == null){throw new System.InvalidOperationException();}if (this._enclosing.remove(this.lastReturned.key);this.lastReturned = null;}
public virtual int lastIndexOf(object @object){if (@object != null){{for (int i = a.Length - 1; i >= 0; i--){if (@object.Equals(a[i])){return i;}}}else{{for (int i = a.Length - 1; i >= 0; i--){if ((object)a[i] == null){return i;}}}}return -1;}
