	mov	(1|M0)	r96.0<1>:uq	r1.0<0;1,0>:uq					
	mov	(1|M0)	r97.0<1>:uq	r1.1<0;1,0>:uq					
	mov	(1|M0)	r98.0<1>:uq	r1.2<0;1,0>:uq					
	mul	(1|M0)	r99.0<1>:ud	r0.1<0;1,0>:ud	0x200:uw				
	mov	(1|M0)	r99.1<1>:ud	0x0:uw					
	add	(1|M0)	r96.0<1>:uq	r96.0<0;1,0>:uq	r99.0<0;1,0>:uq				
	mul	(1|M0)	r99.0<1>:ud	r0.1<0;1,0>:ud	0x400:uw				
	mov	(1|M0)	r99.1<1>:ud	0x0:uw					
	add	(1|M0)	r98.0<1>:uq	r98.0<0;1,0>:uq	r99.0<0;1,0>:uq				
	mov	(8|M0)	r101.0<1>:uw	0x76543210:uv					
	mov	(8|M0)	r101.8<1>:uw	0xfedcba98:uv					
	mov	(16|M0)	r102.0<2>:uw	r101.0<16;16,1>:uw					
	mov	(16|M0)	r102.1<2>:uw	0x0:uw					
	shl	(16|M0)	r102.0<1>:ud	r102.0<8;8,1>:ud	0x6:w				
	mov	(16|M0)	r103.0<1>:ud	r102.0<8;8,1>:ud					
	shr	(16|M0)	r102.0<1>:ud	r102.0<8;8,1>:ud	0x1:w				
	mov	(8|M0)	r108.0<2>:ud	r102.0<8;8,1>:ud					
	mov	(8|M0)	r108.1<2>:ud	0x0:uw					
	mov	(8|M0)	r109.0<2>:ud	r102.8<8;8,1>:ud					
	mov	(8|M0)	r109.1<2>:ud	0x0:uw					
	mov	(8|M0)	r112.0<2>:ud	r103.0<8;8,1>:ud					
	mov	(8|M0)	r112.1<2>:ud	0x0:uw					
	mov	(8|M0)	r113.0<2>:ud	r103.8<8;8,1>:ud					
	mov	(8|M0)	r113.1<2>:ud	0x0:uw					
	add	(8|M0)	r108.0<1>:uq	r108.0<4;4,1>:uq	r96.0<0;1,0>:uq				
	add	(8|M0)	r109.0<1>:uq	r109.0<4;4,1>:uq	r96.0<0;1,0>:uq				
	add	(8|M0)	r112.0<1>:uq	r112.0<4;4,1>:uq	r98.0<0;1,0>:uq				
	add	(8|M0)	r113.0<1>:uq	r113.0<4;4,1>:uq	r98.0<0;1,0>:uq				
	add	(1|M0)	r96.0<1>:uq	r96.0<0;1,0>:uq	0x200:uw				
	add	(1|M0)	r98.0<1>:uq	r98.0<0;1,0>:uq	0x400:uw				
	send.ugm	(1|M0)	r24	r97	null:0	0x0	0x210d580		
	send.ugm	(1|M0)	r25	r97	null:0	0x40000	0x210d580		
	send.ugm	(1|M0)	r26	r97	null:0	0x80000	0x210d580		
	send.ugm	(1|M0)	r27	r97	null:0	0xc0000	0x210d580		
	send.ugm	(1|M0)	r28	r97	null:0	0x100000	0x210d580		
	send.ugm	(1|M0)	r29	r97	null:0	0x140000	0x210d580		
	send.ugm	(1|M0)	r30	r97	null:0	0x180000	0x210d580		
	send.ugm	(1|M0)	r31	r97	null:0	0x1c0000	0x210d580		
	send.ugm	(16|M0)	r16	r108	null:0	0x0	0x4100580		
	send.ugm	(16|M0)	r17	r108	null:0	0x4000	0x4100580		
	send.ugm	(16|M0)	r18	r108	null:0	0x8000	0x4100580		
	send.ugm	(16|M0)	r19	r108	null:0	0xc000	0x4100580		
	send.ugm	(16|M0)	r20	r108	null:0	0x10000	0x4100580		
	send.ugm	(16|M0)	r21	r108	null:0	0x14000	0x4100580		
	send.ugm	(16|M0)	r22	r108	null:0	0x18000	0x4100580		
	send.ugm	(16|M0)	r23	r108	null:0	0x1c000	0x4100580		
	mov	(16|M0)	r32.0<1>:d	0x0:w					
	mov	(16|M0)	r33.0<1>:d	0x0:w					
	mov	(16|M0)	r34.0<1>:d	0x0:w					
	mov	(16|M0)	r35.0<1>:d	0x0:w					
	mov	(16|M0)	r36.0<1>:d	0x0:w					
	mov	(16|M0)	r37.0<1>:d	0x0:w					
	mov	(16|M0)	r38.0<1>:d	0x0:w					
	mov	(16|M0)	r39.0<1>:d	0x0:w					
	mov	(16|M0)	r40.0<1>:d	0x0:w					
	mov	(16|M0)	r41.0<1>:d	0x0:w					
	mov	(16|M0)	r42.0<1>:d	0x0:w					
	mov	(16|M0)	r43.0<1>:d	0x0:w					
	mov	(16|M0)	r44.0<1>:d	0x0:w					
	mov	(16|M0)	r45.0<1>:d	0x0:w					
	mov	(16|M0)	r46.0<1>:d	0x0:w					
	mov	(16|M0)	r47.0<1>:d	0x0:w					
	mov	(1|M0)	r100.0<1>:ud	r1.6<0;1,0>:ud					
L0:
	dpas.8x8	(16|M0)	r32.0:d	r32.0:d	r16.0:b	r24.0:b			
	dpas.8x8	(16|M0)	r40.0:d	r40.0:d	r16.0:b	r28.0:b			
	add	(1|M0)	r100.0<1>:d	r100.0<0;1,0>:d	0xffff:w				
	cmp	(1|M0)	(gt)f0.0	null:d	r100.0<0;1,0>:d	0x0:w				
(W&f0.0)	jmpi	(1|M0)		L0					
	send.ugm	(16|M0)	null	r112	r32:1	0x0	0x4000584		
	send.ugm	(16|M0)	null	r112	r33:1	0x4000	0x4000584		
	send.ugm	(16|M0)	null	r112	r34:1	0x8000	0x4000584		
	send.ugm	(16|M0)	null	r112	r35:1	0xc000	0x4000584		
	send.ugm	(16|M0)	null	r112	r36:1	0x10000	0x4000584		
	send.ugm	(16|M0)	null	r112	r37:1	0x14000	0x4000584		
	send.ugm	(16|M0)	null	r112	r38:1	0x18000	0x4000584		
	send.ugm	(16|M0)	null	r112	r39:1	0x1c000	0x4000584		
	send.ugm	(16|M0)	null	r112	r40:1	0x20000	0x4000584		
	send.ugm	(16|M0)	null	r112	r41:1	0x24000	0x4000584		
	send.ugm	(16|M0)	null	r112	r42:1	0x28000	0x4000584		
	send.ugm	(16|M0)	null	r112	r43:1	0x2c000	0x4000584		
	send.ugm	(16|M0)	null	r112	r44:1	0x30000	0x4000584		
	send.ugm	(16|M0)	null	r112	r45:1	0x34000	0x4000584		
	send.ugm	(16|M0)	null	r112	r46:1	0x38000	0x4000584		
	send.ugm	(16|M0)	null	r112	r47:1	0x3c000	0x4000584		
