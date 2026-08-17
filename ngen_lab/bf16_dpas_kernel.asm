	mov	(1|M0)	r96.0<1>:uq	r1.0<0;1,0>:uq					
	mov	(1|M0)	r97.0<1>:uq	r1.1<0;1,0>:uq					
	mov	(1|M0)	r98.0<1>:uq	r1.2<0;1,0>:uq					
	mul	(1|M0)	r99.0<1>:ud	r0.1<0;1,0>:ud	0x400:uw				
	mov	(1|M0)	r99.1<1>:ud	0x0:uw					
	add	(1|M0)	r96.0<1>:uq	r96.0<0;1,0>:uq	r99.0<0;1,0>:uq				
	add	(1|M0)	r98.0<1>:uq	r98.0<0;1,0>:uq	r99.0<0;1,0>:uq				
	mov	(8|M0)	r101.0<1>:uw	0x76543210:uv					
	mov	(8|M0)	r101.8<1>:uw	0xfedcba98:uv					
	mov	(16|M0)	r102.0<2>:uw	r101.0<16;16,1>:uw					
	mov	(16|M0)	r102.1<2>:uw	0x0:uw					
	shl	(16|M0)	r102.0<1>:ud	r102.0<8;8,1>:ud	0x6:w				
	mov	(8|M0)	r104.0<2>:ud	r102.0<8;8,1>:ud					
	mov	(8|M0)	r104.1<2>:ud	0x0:uw					
	mov	(8|M0)	r105.0<2>:ud	r102.8<8;8,1>:ud					
	mov	(8|M0)	r105.1<2>:ud	0x0:uw					
	mov	(8|M0)	r106.0<2>:ud	r102.0<8;8,1>:ud					
	mov	(8|M0)	r106.1<2>:ud	0x0:uw					
	mov	(8|M0)	r107.0<2>:ud	r102.8<8;8,1>:ud					
	mov	(8|M0)	r107.1<2>:ud	0x0:uw					
	add	(8|M0)	r104.0<1>:uq	r104.0<4;4,1>:uq	r96.0<0;1,0>:uq				
	add	(8|M0)	r105.0<1>:uq	r105.0<4;4,1>:uq	r96.0<0;1,0>:uq				
	add	(8|M0)	r106.0<1>:uq	r106.0<4;4,1>:uq	r98.0<0;1,0>:uq				
	add	(8|M0)	r107.0<1>:uq	r107.0<4;4,1>:uq	r98.0<0;1,0>:uq				
	send.ugm	(1|M0)	r32	r97	null:0	0x0	0x210d580		
	send.ugm	(1|M0)	r33	r97	null:0	0x40000	0x210d580		
	send.ugm	(1|M0)	r34	r97	null:0	0x80000	0x210d580		
	send.ugm	(1|M0)	r35	r97	null:0	0xc0000	0x210d580		
	send.ugm	(1|M0)	r36	r97	null:0	0x100000	0x210d580		
	send.ugm	(1|M0)	r37	r97	null:0	0x140000	0x210d580		
	send.ugm	(1|M0)	r38	r97	null:0	0x180000	0x210d580		
	send.ugm	(1|M0)	r39	r97	null:0	0x1c0000	0x210d580		
	send.ugm	(1|M0)	r40	r97	null:0	0x200000	0x210d580		
	send.ugm	(1|M0)	r41	r97	null:0	0x240000	0x210d580		
	send.ugm	(1|M0)	r42	r97	null:0	0x280000	0x210d580		
	send.ugm	(1|M0)	r43	r97	null:0	0x2c0000	0x210d580		
	send.ugm	(1|M0)	r44	r97	null:0	0x300000	0x210d580		
	send.ugm	(1|M0)	r45	r97	null:0	0x340000	0x210d580		
	send.ugm	(1|M0)	r46	r97	null:0	0x380000	0x210d580		
	send.ugm	(1|M0)	r47	r97	null:0	0x3c0000	0x210d580		
	send.ugm	(16|M0)	r16	r104	null:0	0x0	0x4100580		
	send.ugm	(16|M0)	r17	r104	null:0	0x4000	0x4100580		
	send.ugm	(16|M0)	r18	r104	null:0	0x8000	0x4100580		
	send.ugm	(16|M0)	r19	r104	null:0	0xc000	0x4100580		
	send.ugm	(16|M0)	r20	r104	null:0	0x10000	0x4100580		
	send.ugm	(16|M0)	r21	r104	null:0	0x14000	0x4100580		
	send.ugm	(16|M0)	r22	r104	null:0	0x18000	0x4100580		
	send.ugm	(16|M0)	r23	r104	null:0	0x1c000	0x4100580		
	send.ugm	(16|M0)	r24	r104	null:0	0x20000	0x4100580		
	send.ugm	(16|M0)	r25	r104	null:0	0x24000	0x4100580		
	send.ugm	(16|M0)	r26	r104	null:0	0x28000	0x4100580		
	send.ugm	(16|M0)	r27	r104	null:0	0x2c000	0x4100580		
	send.ugm	(16|M0)	r28	r104	null:0	0x30000	0x4100580		
	send.ugm	(16|M0)	r29	r104	null:0	0x34000	0x4100580		
	send.ugm	(16|M0)	r30	r104	null:0	0x38000	0x4100580		
	send.ugm	(16|M0)	r31	r104	null:0	0x3c000	0x4100580		
	mov	(16|M0)	r48.0<1>:f	0x0:f					
	mov	(16|M0)	r49.0<1>:f	0x0:f					
	mov	(16|M0)	r50.0<1>:f	0x0:f					
	mov	(16|M0)	r51.0<1>:f	0x0:f					
	mov	(16|M0)	r52.0<1>:f	0x0:f					
	mov	(16|M0)	r53.0<1>:f	0x0:f					
	mov	(16|M0)	r54.0<1>:f	0x0:f					
	mov	(16|M0)	r55.0<1>:f	0x0:f					
	mov	(16|M0)	r56.0<1>:f	0x0:f					
	mov	(16|M0)	r57.0<1>:f	0x0:f					
	mov	(16|M0)	r58.0<1>:f	0x0:f					
	mov	(16|M0)	r59.0<1>:f	0x0:f					
	mov	(16|M0)	r60.0<1>:f	0x0:f					
	mov	(16|M0)	r61.0<1>:f	0x0:f					
	mov	(16|M0)	r62.0<1>:f	0x0:f					
	mov	(16|M0)	r63.0<1>:f	0x0:f					
	mov	(1|M0)	r100.0<1>:ud	r1.6<0;1,0>:ud					
L0:
	dpas.8x8	(16|M0)	r48.0:f	r48.0:f	r16.0:bf	r32.0:bf			
	dpas.8x8	(16|M0)	r48.0:f	r48.0:f	r24.0:bf	r40.0:bf			
	dpas.8x8	(16|M0)	r56.0:f	r56.0:f	r16.0:bf	r36.0:bf			
	dpas.8x8	(16|M0)	r56.0:f	r56.0:f	r24.0:bf	r44.0:bf			
	add	(1|M0)	r100.0<1>:d	r100.0<0;1,0>:d	0xffff:w				
	cmp	(1|M0)	(gt)f0.0	null:d	r100.0<0;1,0>:d	0x0:w				
(W&f0.0)	jmpi	(1|M0)		L0					
	send.ugm	(16|M0)	null	r106	r48:1	0x0	0x4000584		
	send.ugm	(16|M0)	null	r106	r49:1	0x4000	0x4000584		
	send.ugm	(16|M0)	null	r106	r50:1	0x8000	0x4000584		
	send.ugm	(16|M0)	null	r106	r51:1	0xc000	0x4000584		
	send.ugm	(16|M0)	null	r106	r52:1	0x10000	0x4000584		
	send.ugm	(16|M0)	null	r106	r53:1	0x14000	0x4000584		
	send.ugm	(16|M0)	null	r106	r54:1	0x18000	0x4000584		
	send.ugm	(16|M0)	null	r106	r55:1	0x1c000	0x4000584		
	send.ugm	(16|M0)	null	r106	r56:1	0x20000	0x4000584		
	send.ugm	(16|M0)	null	r106	r57:1	0x24000	0x4000584		
	send.ugm	(16|M0)	null	r106	r58:1	0x28000	0x4000584		
	send.ugm	(16|M0)	null	r106	r59:1	0x2c000	0x4000584		
	send.ugm	(16|M0)	null	r106	r60:1	0x30000	0x4000584		
	send.ugm	(16|M0)	null	r106	r61:1	0x34000	0x4000584		
	send.ugm	(16|M0)	null	r106	r62:1	0x38000	0x4000584		
	send.ugm	(16|M0)	null	r106	r63:1	0x3c000	0x4000584		
